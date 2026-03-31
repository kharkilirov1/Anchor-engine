from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.anchor_detector import AnchorDetector
from src.model.anchor_memory import AnchorMemory
from src.model.anchor_monitor import ContradictionMonitor
from src.model.anchor_types import AnchorRecord, RevisionDecision
from src.model.anchor_tree_types import ProposalRepairDiagnostics
from src.model.anchor_revision import RevisionController
from src.model.anchor_viability import ViabilityTracker
from src.model.config import ModelConfig, TOY_CONFIG
from src.model.future_influence import FutureInfluenceScorer
from src.model.qwen_generation_bias import (
    apply_frequency_penalty,
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    compute_entropy_conflict_bias_scale,
    compute_anchor_logits_bias,
    compute_normalized_entropy,
)
from src.model.future_span_hints import (
    build_auxiliary_future_proposals,
    build_future_hint_candidates,
    compute_span_anchor_overlap,
    decode_span_text,
    extract_high_influence_spans,
    summarize_auxiliary_proposals,
)
from src.model.anchor_tree_builder import build_observed_tree
from src.model.anchor_tree_match import greedy_tree_match
from src.model.anchor_tree_templates import get_expected_tree_template
from src.model.anchor_tree_consistency import compute_graph_consistency, compute_tree_consistency
from src.model.anchor_tree_proposals import rank_proposals_by_tree_repair


class QwenAnchorOverlay(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        cfg: ModelConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        model_cfg = getattr(base_model, "config", None)
        if model_cfg is None:
            raise ValueError("base_model must expose a `.config` object")

        hidden_size = int(getattr(model_cfg, "hidden_size"))
        vocab_size = int(getattr(model_cfg, "vocab_size"))
        max_seq_len = int(
            getattr(
                model_cfg,
                "max_position_embeddings",
                getattr(model_cfg, "max_seq_len", 32768),
            )
        )

        anchor_cfg = replace(
            cfg or TOY_CONFIG,
            d_model=hidden_size,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        self.cfg = anchor_cfg
        self.anchor_detector = AnchorDetector(anchor_cfg)
        self.anchor_memory = AnchorMemory(anchor_cfg)
        self.contradiction_monitor = ContradictionMonitor(anchor_cfg)
        self.viability_tracker = ViabilityTracker(anchor_cfg)
        self.revision_controller = RevisionController(anchor_cfg)
        self.future_influence_scorer = FutureInfluenceScorer()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        cfg: ModelConfig | None = None,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "QwenAnchorOverlay":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is required for QwenAnchorOverlay.from_pretrained"
            ) from exc

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        overlay = cls(base_model=model, cfg=cfg, tokenizer=tokenizer)
        if device is not None:
            overlay = overlay.to(device)
        return overlay

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        anchor_out = self.analyze_hidden_batch(
            hidden=hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        anchor_out["logits"] = outputs.logits
        anchor_out["hidden"] = hidden
        return anchor_out

    def analyze_hidden_batch(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        detector_out, anchors, contradiction_out, viability_out = self._prepare_anchor_state(
            hidden=hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        revised_anchors, revision_events, diagnostics = self._apply_revision_path(
            anchors=self._clone_anchor_batches(anchors),
            viability_out=viability_out,
            arbiter={},
        )
        active_anchors = self.anchor_memory.get_active_anchors(revised_anchors)
        diagnostics["revision_event_count"] = len(revision_events)

        return {
            "anchor_candidates": detector_out["candidates"],
            "active_anchors": active_anchors,
            "anchor_states": [[anchor.state for anchor in batch] for batch in revised_anchors],
            "contradiction_pressure": contradiction_out["contradiction_pressure"],
            "viability": viability_out["viability"],
            "revision_events": revision_events,
            "anchor_diagnostics": diagnostics,
            "detector_scores": detector_out["scores"],
            "proposal_diagnostics": {
                "proposal_count": 0,
                "regime_shift_count": 0,
                "anchors_with_proposal_influence": 0,
                "mean_proposal_score": 0.0,
                "mean_blend_ratio": 0.0,
            },
            "_pre_revision_anchors": anchors,
            "_viability_out": viability_out,
            "_contradiction_out": contradiction_out,
        }

    def _prepare_anchor_state(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], list[list[AnchorRecord]], dict[str, Any], dict[str, Any]]:
        anchor_dtype = self.anchor_detector.prior_head.weight.dtype
        anchor_hidden = hidden if hidden.dtype == anchor_dtype else hidden.to(anchor_dtype)
        history = torch.roll(anchor_hidden.detach(), shifts=1, dims=1)
        history[:, 0] = anchor_hidden[:, 0].detach()

        detector_out = self.anchor_detector(anchor_hidden, history, attention_mask=attention_mask)
        anchors = self.anchor_memory.add_candidates(detector_out["candidates"])
        anchors = self.anchor_memory.update_support(anchors, detector_out["scores"])
        anchors = self.anchor_memory.update_ttl(anchors)

        contradiction_out = self.contradiction_monitor(
            anchor_hidden,
            anchors,
            aux={"input_ids": input_ids},
        )
        viability_out = self.viability_tracker(anchors, contradiction_out)
        return detector_out, anchors, contradiction_out, viability_out

    def _apply_revision_path(
        self,
        anchors: list[list[AnchorRecord]],
        viability_out: dict[str, Any],
        arbiter: dict[int, dict[str, Any]],
    ) -> tuple[list[list[AnchorRecord]], list[RevisionDecision], dict[str, Any]]:
        revision_events = self.revision_controller(anchors, viability_out, arbiter=arbiter)
        revised_anchors = self.anchor_memory.apply_revision(anchors, revision_events)
        diagnostics = self.anchor_memory.export_diagnostics(revised_anchors)
        return revised_anchors, revision_events, diagnostics

    @staticmethod
    def _clone_anchor_batches(anchors: list[list[AnchorRecord]]) -> list[list[AnchorRecord]]:
        return [
            [
                AnchorRecord(
                    id=anchor.id,
                    start_idx=anchor.start_idx,
                    end_idx=anchor.end_idx,
                    repr=anchor.repr.detach().clone(),
                    score=float(anchor.score.detach().item()) if isinstance(anchor.score, torch.Tensor) else float(anchor.score),
                    state=anchor.state,
                    support=float(anchor.support.detach().item()) if isinstance(anchor.support, torch.Tensor) else float(anchor.support),
                    contradiction_pressure=float(anchor.contradiction_pressure.detach().item())
                    if isinstance(anchor.contradiction_pressure, torch.Tensor)
                    else float(anchor.contradiction_pressure),
                    viability=float(anchor.viability.detach().item()) if isinstance(anchor.viability, torch.Tensor) else float(anchor.viability),
                    ttl=float(anchor.ttl.detach().item()) if isinstance(anchor.ttl, torch.Tensor) else float(anchor.ttl),
                    parent_id=anchor.parent_id,
                    branch_id=anchor.branch_id,
                    descendant_mass=float(anchor.descendant_mass.detach().item())
                    if isinstance(anchor.descendant_mass, torch.Tensor)
                    else (0.0 if anchor.descendant_mass is None else float(anchor.descendant_mass)),
                    descendant_coherence=float(anchor.descendant_coherence.detach().item())
                    if isinstance(anchor.descendant_coherence, torch.Tensor)
                    else (0.0 if anchor.descendant_coherence is None else float(anchor.descendant_coherence)),
                )
                for anchor in batch
            ]
            for batch in anchors
        ]

    def _build_auxiliary_arbiter(
        self,
        anchors: list[list[AnchorRecord]],
        auxiliary_proposal_batches: list[list[dict[str, Any]]],
        proposal_repair_batches: list[list[ProposalRepairDiagnostics]] | None = None,
    ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
        arbiter: dict[int, dict[str, Any]] = {}
        batch_summaries: list[dict[str, Any]] = []
        repair_batches = proposal_repair_batches or [[] for _ in auxiliary_proposal_batches]
        for batch_anchors, proposals, repair_diagnostics in zip(anchors, auxiliary_proposal_batches, repair_batches):
            matches = 0
            alt_prob_sum = 0.0
            score_sum = 0.0
            repair_gain_sum = 0.0
            repair_map = {
                str(item.notes.get("proposal_text", "")): item
                for item in repair_diagnostics
            }
            candidate_edges: list[tuple[float, AnchorRecord, dict[str, Any], float]] = []
            for anchor in batch_anchors:
                for proposal in proposals:
                    start, end = proposal["proposal_span"]
                    if start <= int(anchor.end_idx):
                        continue
                    repair_item = repair_map.get(str(proposal.get("proposal_text", "")))
                    repair_gain = 0.0 if repair_item is None else float(repair_item.repair_gain)
                    proposal_with_repair = dict(proposal)
                    if repair_item is not None:
                        proposal_with_repair["repair_gain"] = repair_gain
                    gate = self._auxiliary_match_gate(anchor=anchor, proposal=proposal_with_repair)
                    if gate <= 0.20:
                        continue
                    candidate_edges.append((gate, anchor, proposal_with_repair, repair_gain))

            candidate_edges.sort(key=lambda item: item[0], reverse=True)
            used_anchor_ids: set[int] = set()
            used_proposal_spans: set[tuple[int, int]] = set()
            for gate, anchor, best_match, repair_gain in candidate_edges:
                proposal_span = tuple(int(x) for x in best_match["proposal_span"])
                if anchor.id in used_anchor_ids or proposal_span in used_proposal_spans:
                    continue
                used_anchor_ids.add(anchor.id)
                used_proposal_spans.add(proposal_span)
                alt_prob = max(0.0, min(0.95, gate))
                arbiter[anchor.id] = {
                    "prefer_current_prob": 1.0 - alt_prob,
                    "prefer_alt_prob": alt_prob,
                    "proposal_score": float(best_match.get("proposal_score", 0.0)),
                    "proposal_root_token": best_match.get("proposal_root_token"),
                    "proposal_type": best_match.get("proposal_type", "future_hint_span"),
                    "proposal_span": proposal_span,
                    "proposal_text": best_match.get("proposal_text"),
                    "distance_from_anchor": int(proposal_span[0]) - int(anchor.end_idx),
                    "repair_gain": float(repair_gain),
                }
                matches += 1
                alt_prob_sum += alt_prob
                score_sum += float(best_match.get("proposal_score", 0.0))
                repair_gain_sum += float(repair_gain)
            batch_summaries.append(
                {
                    "matched_anchor_count": matches,
                    "mean_alt_prob": alt_prob_sum / max(matches, 1),
                    "mean_matched_proposal_score": score_sum / max(matches, 1),
                    "mean_repair_gain": repair_gain_sum / max(matches, 1),
                }
            )
        return arbiter, batch_summaries

    @staticmethod
    def _auxiliary_match_gate(anchor: AnchorRecord, proposal: dict[str, Any]) -> float:
        start = int(proposal["proposal_span"][0])
        distance = max(1, start - int(anchor.end_idx))
        distance_penalty = 1.0 / float(distance) ** 0.5
        proposal_score = float(proposal.get("proposal_score", 0.0))
        repair_gain = float(proposal.get("repair_gain", 0.0))
        pressure = float(anchor.contradiction_pressure)
        viability = float(anchor.viability)
        pressure_factor = 0.20 + 0.80 * pressure
        viability_factor = 0.55 + 0.45 * viability
        repair_factor = 1.0 if "repair_gain" not in proposal else min(2.0, max(0.05, 0.5 + repair_gain))
        return proposal_score * distance_penalty * pressure_factor * viability_factor * repair_factor

    @staticmethod
    def _summarize_auxiliary_revision(
        base_events: list[RevisionDecision],
        auxiliary_events: list[RevisionDecision],
        batch_summaries: list[dict[str, Any]],
    ) -> dict[str, float]:
        def _count(events: list[RevisionDecision], action: str) -> int:
            return sum(1 for event in events if event.action == action)

        matched_counts = [item["matched_anchor_count"] for item in batch_summaries]
        alt_probs = [item["mean_alt_prob"] for item in batch_summaries if item["matched_anchor_count"] > 0]
        matched_scores = [
            item["mean_matched_proposal_score"] for item in batch_summaries if item["matched_anchor_count"] > 0
        ]
        repair_gains = [item["mean_repair_gain"] for item in batch_summaries if item["matched_anchor_count"] > 0]
        base_revise = _count(base_events, "revise")
        auxiliary_revise = _count(auxiliary_events, "revise")
        base_retire = _count(base_events, "retire")
        auxiliary_retire = _count(auxiliary_events, "retire")
        return {
            "matched_anchor_count": int(sum(matched_counts)),
            "batches_with_matches": int(sum(1 for count in matched_counts if count > 0)),
            "mean_alt_prob": float(sum(alt_probs) / max(len(alt_probs), 1)) if alt_probs else 0.0,
            "mean_matched_proposal_score": (
                float(sum(matched_scores) / max(len(matched_scores), 1)) if matched_scores else 0.0
            ),
            "mean_repair_gain": float(sum(repair_gains) / max(len(repair_gains), 1)) if repair_gains else 0.0,
            "base_revise_count": int(base_revise),
            "auxiliary_revise_count": int(auxiliary_revise),
            "auxiliary_revise_gain": int(auxiliary_revise - base_revise),
            "base_retire_count": int(base_retire),
            "auxiliary_retire_count": int(auxiliary_retire),
            "auxiliary_retire_delta": int(auxiliary_retire - base_retire),
        }

    def extract_hidden_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def analyze_texts(
        self,
        texts: list[str],
        max_length: int = 256,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for analyze_texts")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        batch = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self.extract_hidden_batch(**batch)
            out = self.analyze_hidden_batch(
                hidden=hidden,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            out["hidden"] = hidden
        return out, batch

    def analyze_texts_with_future_influence(
        self,
        texts: list[str],
        max_length: int = 256,
        future_window: int = 16,
        span_threshold: float = 0.75,
        top_spans: int = 5,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for analyze_texts_with_future_influence")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        batch = {key: value.to(device) for key, value in encoded.items()}
        out = self(**batch)
        out["future_influence"] = self.future_influence_scorer(
            hidden=out["hidden"],
            logits=out["logits"],
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            future_window=future_window,
        )
        hint_batches: list[dict[str, Any]] = []
        auxiliary_proposal_batches: list[list[dict[str, Any]]] = []
        observed_tree_batches: list[dict[str, Any]] = []
        scores = out["future_influence"]["scores"]
        masks = batch.get("attention_mask")
        for batch_idx in range(batch["input_ids"].size(0)):
            valid_len = (
                int(masks[batch_idx].sum().item())
                if masks is not None
                else int(batch["input_ids"][batch_idx].numel())
            )
            trimmed_scores = scores[batch_idx, :valid_len]
            trimmed_ids = batch["input_ids"][batch_idx, :valid_len]
            active_anchor_spans = [
                {
                    "start": max(0, min(int(anchor.start_idx), valid_len - 1)),
                    "end": max(0, min(int(anchor.end_idx), valid_len - 1)),
                }
                for anchor in out["active_anchors"][batch_idx]
                if valid_len > 0
            ]
            future_spans = extract_high_influence_spans(
                scores=trimmed_scores,
                input_ids=trimmed_ids,
                tokenizer=self.tokenizer,
                min_score=span_threshold,
                top_spans=top_spans,
            )
            future_hint_candidates = build_future_hint_candidates(future_spans, active_anchor_spans)
            overlap = compute_span_anchor_overlap(future_spans, active_anchor_spans)
            auxiliary_proposals = build_auxiliary_future_proposals(
                hidden=out["hidden"][batch_idx, :valid_len],
                input_ids=trimmed_ids,
                future_hint_candidates=future_hint_candidates,
                tokenizer=self.tokenizer,
            )
            auxiliary_proposal_batches.append(auxiliary_proposals)
            active_anchor_payloads = [
                {
                    "anchor": anchor,
                    "text": decode_span_text(
                        self.tokenizer,
                        [int(token.item()) for token in trimmed_ids[max(0, int(anchor.start_idx)) : min(valid_len, int(anchor.end_idx) + 1)]],
                    ),
                    "start": max(0, min(int(anchor.start_idx), valid_len - 1)),
                    "end": max(0, min(int(anchor.end_idx), valid_len - 1)),
                }
                for anchor in out["active_anchors"][batch_idx]
                if valid_len > 0
            ]
            base_observed_tree = build_observed_tree(
                text=texts[batch_idx],
                active_anchors=active_anchor_payloads,
                future_hint_candidates=future_hint_candidates,
                auxiliary_proposals=[],
            )
            observed_tree = build_observed_tree(
                text=texts[batch_idx],
                active_anchors=active_anchor_payloads,
                future_hint_candidates=future_hint_candidates,
                auxiliary_proposals=auxiliary_proposals,
                domain=base_observed_tree.domain,
            )
            tree_match = None
            tree_diagnostics = None
            proposal_repair = []
            if observed_tree.domain != "unknown":
                expected_tree = get_expected_tree_template(observed_tree.domain)
                tree_match = greedy_tree_match(observed_tree, expected_tree)
                consistency = compute_tree_consistency(observed_tree, expected_tree)
                proposal_repair = rank_proposals_by_tree_repair(
                    current_tree=base_observed_tree,
                    expected_tree=expected_tree,
                    proposal_candidates=auxiliary_proposals,
                )
                tree_diagnostics = {
                    "domain": observed_tree.domain,
                    "coverage": float(consistency.coverage),
                    "spurious_ratio": float(consistency.spurious_ratio),
                    "alignment_score": float(consistency.alignment_score),
                    "missing_required_count": int(consistency.missing_required_count),
                    "order_violations": int(consistency.order_violations),
                    "drift_score": float(consistency.drift_score),
                    "repair_candidate_count": int(len(proposal_repair)),
                    "best_repair_gain": float(proposal_repair[0].repair_gain) if proposal_repair else 0.0,
                }
            observed_tree_batches.append(
                {
                    "domain": observed_tree.domain,
                    "base_observed_tree": base_observed_tree,
                    "observed_tree": observed_tree,
                    "tree_match": tree_match,
                    "tree_diagnostics": tree_diagnostics,
                    "proposal_repair": proposal_repair,
                }
            )
            hint_batches.append(
                {
                    "active_anchor_spans": active_anchor_spans,
                    "future_spans": future_spans,
                    "future_hint_candidates": future_hint_candidates,
                    "auxiliary_proposals": auxiliary_proposals,
                    "tree_domain": observed_tree.domain,
                    "tree_diagnostics": tree_diagnostics,
                    **overlap,
                }
            )
        out["future_hint_batches"] = hint_batches
        out["observed_tree_batches"] = observed_tree_batches
        observed_trees = [batch_item["observed_tree"] for batch_item in observed_tree_batches]
        out["observed_tree_graph_diagnostics"] = compute_graph_consistency(observed_trees)
        out["auxiliary_proposal_batches"] = auxiliary_proposal_batches
        out["auxiliary_proposal_diagnostics"] = summarize_auxiliary_proposals(auxiliary_proposal_batches)
        auxiliary_arbiter, auxiliary_batch_summaries = self._build_auxiliary_arbiter(
            anchors=self._clone_anchor_batches(out["_pre_revision_anchors"]),
            auxiliary_proposal_batches=auxiliary_proposal_batches,
            proposal_repair_batches=[batch_item["proposal_repair"] for batch_item in observed_tree_batches],
        )
        auxiliary_revised_anchors, auxiliary_revision_events, auxiliary_anchor_diagnostics = self._apply_revision_path(
            anchors=self._clone_anchor_batches(out["_pre_revision_anchors"]),
            viability_out=out["_viability_out"],
            arbiter=auxiliary_arbiter,
        )
        out["auxiliary_revision_events"] = auxiliary_revision_events
        out["auxiliary_revision_anchor_diagnostics"] = auxiliary_anchor_diagnostics
        out["auxiliary_revision_batch_summaries"] = auxiliary_batch_summaries
        out["auxiliary_revision_diagnostics"] = self._summarize_auxiliary_revision(
            base_events=out["revision_events"],
            auxiliary_events=auxiliary_revision_events,
            batch_summaries=auxiliary_batch_summaries,
        )
        out["auxiliary_active_anchors"] = self.anchor_memory.get_active_anchors(auxiliary_revised_anchors)
        return out, batch

    def generate_with_anchor_bias(
        self,
        prompt: str,
        max_new_tokens: int = 24,
        max_length: int = 256,
        conflict_threshold: float = 0.55,
        bias_scale: float = 1.50,
        temperature: float = 1.0,
        greedy: bool = True,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.05,
        no_repeat_ngram_size: int = 3,
        max_bias_gate_sum: float = 1.50,
        entropy_top_k: int = 32,
        entropy_threshold: float = 0.40,
        entropy_slope: float = 0.08,
        pressure_threshold: float | None = None,
        pressure_slope: float = 0.08,
        pressure_rescue_floor: float = 0.20,
        min_bias_pressure: float | None = None,
    ) -> dict[str, Any]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for generate_with_anchor_bias")

        encoded = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        generated = input_ids
        generated_mask = attention_mask
        step_records: list[dict[str, Any]] = []
        output_projection = self.base_model.get_output_embeddings()
        if output_projection is None:
            raise ValueError("base model must expose output embeddings")

        for _ in range(max_new_tokens):
            outputs = self.base_model(
                input_ids=generated,
                attention_mask=generated_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            next_hidden = hidden[:, -1, :]
            next_logits = outputs.logits[:, -1, :]

            detector_out, anchors, contradiction_out, viability_out = self._prepare_anchor_state(
                hidden=hidden,
                input_ids=generated,
                attention_mask=generated_mask,
            )
            revised_anchors, revision_events, diagnostics = self._apply_revision_path(
                anchors=self._clone_anchor_batches(anchors),
                viability_out=viability_out,
                arbiter={},
            )
            active_anchors = self.anchor_memory.get_active_anchors(revised_anchors)[0]
            mean_pressure = float(diagnostics["mean_contradiction_pressure"])
            effective_pressure_threshold = (
                float(min_bias_pressure)
                if min_bias_pressure is not None
                else (0.60 if pressure_threshold is None else float(pressure_threshold))
            )
            normalized_entropy = float(compute_normalized_entropy(next_logits, top_k=entropy_top_k).item())
            alpha_diag = compute_entropy_conflict_bias_scale(
                normalized_entropy=normalized_entropy,
                contradiction_pressure=mean_pressure,
                alpha_max=float(bias_scale),
                entropy_threshold=float(entropy_threshold),
                pressure_threshold=effective_pressure_threshold,
                entropy_slope=float(entropy_slope),
                pressure_slope=float(pressure_slope),
                pressure_rescue_floor=float(pressure_rescue_floor),
            )
            bias_diag: list[dict[str, float]]
            raw_anchor_bias, bias_diag = compute_anchor_logits_bias(
                last_hidden=next_hidden,
                active_anchors=active_anchors,
                output_projection=output_projection,
                conflict_threshold=conflict_threshold,
                bias_scale=1.0,
            )

            gate_sum = float(sum(item["gate"] for item in bias_diag))
            if bias_diag and gate_sum > float(max_bias_gate_sum):
                scale = float(max_bias_gate_sum) / max(gate_sum, 1e-6)
                raw_anchor_bias = raw_anchor_bias * scale
                for item in bias_diag:
                    item["gate"] = float(item["gate"]) * scale
                gate_sum = float(sum(item["gate"] for item in bias_diag))

            effective_scale = float(alpha_diag["alpha_t"])
            anchor_bias = raw_anchor_bias * effective_scale
            adjusted_logits = next_logits + anchor_bias.to(next_logits.dtype)
            adjusted_logits = apply_repetition_penalty(
                logits=adjusted_logits,
                generated_ids=generated,
                penalty=repetition_penalty,
            )
            adjusted_logits = apply_frequency_penalty(
                logits=adjusted_logits,
                generated_ids=generated,
                penalty=frequency_penalty,
            )
            adjusted_logits, blocked_tokens = apply_no_repeat_ngram(
                logits=adjusted_logits,
                generated_ids=generated,
                ngram_size=no_repeat_ngram_size,
            )
            if temperature != 1.0:
                adjusted_logits = adjusted_logits / max(float(temperature), 1e-6)

            if greedy:
                next_token = torch.argmax(adjusted_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(adjusted_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            generated_mask = torch.cat(
                [
                    generated_mask,
                    torch.ones((generated_mask.size(0), 1), device=device, dtype=generated_mask.dtype),
                ],
                dim=1,
            )
            step_records.append(
                {
                    "token_id": int(next_token.item()),
                    "token_text": self.tokenizer.decode([int(next_token.item())], skip_special_tokens=False),
                    "num_active": int(diagnostics["num_active"]),
                    "mean_contradiction_pressure": float(diagnostics["mean_contradiction_pressure"]),
                    "mean_viability": float(diagnostics["mean_viability"]),
                    "dead_end_count": int(diagnostics["dead_end_count"]),
                    "revision_event_count": len(revision_events),
                    "bias_nonzero_anchors": len(bias_diag),
                    "bias_gate_sum": gate_sum,
                    "effective_bias_scale": float(effective_scale),
                    "normalized_entropy": float(normalized_entropy),
                    "entropy_gate": float(alpha_diag["entropy_gate"]),
                    "pressure_gate": float(alpha_diag["pressure_gate"]),
                    "blocked_ngram_token_count": int(len(blocked_tokens)),
                    "bias_applied": bool(bias_diag) and effective_scale > 0.0,
                }
            )
            if int(next_token.item()) == int(getattr(self.tokenizer, "eos_token_id", -1)):
                break
            if generated.size(1) >= max_length:
                break

        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        continuation_ids = generated[0, input_ids.size(1) :]
        continuation_text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "continuation_text": continuation_text,
            "steps": step_records,
        }
