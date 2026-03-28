from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.anchor_detector import AnchorDetector
from src.model.anchor_memory import AnchorMemory
from src.model.anchor_monitor import ContradictionMonitor
from src.model.anchor_revision import AnchorArbiter, RevisionController
from src.model.anchor_viability import ViabilityTracker
from src.model.backbone import Backbone
from src.model.config import ModelConfig


class ABPTAnchorV1(nn.Module):
    _ANCHOR_REGIME_ALIAS: dict[int, int] = {
        11: 11,
        13: 11,
        16: 11,
        21: 21,
        22: 21,
        23: 21,
        31: 31,
        32: 31,
        33: 31,
        41: 41,
        42: 41,
        43: 41,
        44: 41,
        51: 51,
        52: 51,
        53: 51,
    }
    _PROPOSAL_GATE_MULTIPLIER: dict[int, float] = {
        12: 1.10,
        24: 1.35,
        34: 1.10,
        45: 1.10,
        54: 1.00,
    }
    _PROPOSAL_ROOT_PRIOR: dict[int, float] = {
        12: 0.85,
        15: 0.45,
        45: 1.00,
        54: 0.45,
    }

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.anchor_detector = AnchorDetector(cfg)
        self.anchor_memory = AnchorMemory(cfg)
        self.contradiction_monitor = ContradictionMonitor(cfg)
        self.viability_tracker = ViabilityTracker(cfg)
        self.anchor_arbiter = AnchorArbiter(cfg)
        self.revision_controller = RevisionController(cfg)
        self.anchor_gate = nn.Linear(cfg.d_model, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        backbone_out = self.backbone(input_ids)
        hidden = backbone_out["hidden"]

        history = torch.roll(hidden.detach(), shifts=1, dims=1)
        history[:, 0] = hidden[:, 0].detach()
        detector_out = self.anchor_detector(hidden, history)

        anchors = self.anchor_memory.add_candidates(detector_out["candidates"])
        anchors = self.anchor_memory.update_support(anchors, detector_out["scores"])
        anchors = self.anchor_memory.update_ttl(anchors)

        contradiction_out = self.contradiction_monitor(hidden, anchors, aux={"input_ids": input_ids})
        viability_out = self.viability_tracker(anchors, contradiction_out)

        arbiter_out: dict[int, dict] = {}
        alternative_proposals: dict[int, dict] = {}
        for batch_idx, batch_anchors in enumerate(anchors):
            seq_hidden = hidden[batch_idx]
            seq_ids = input_ids[batch_idx]
            for anchor in batch_anchors:
                pressure = contradiction_out["contradiction_pressure"].get(anchor.id, 0.0)
                if pressure < self.cfg.anchor_dead_end_threshold:
                    continue
                proposal = self._propose_alternative_reading(
                    seq_hidden=seq_hidden,
                    seq_ids=seq_ids,
                    anchor=anchor,
                )
                alternative_proposals[anchor.id] = proposal
                arbiter_result = self.anchor_arbiter(
                    seq_hidden,
                    anchor,
                    alt={"repr": proposal["repr"]},
                )
                arbiter_out[anchor.id] = {
                    **arbiter_result,
                    "proposal_score": float(proposal.get("proposal_score", 0.0)),
                    "proposal_root_token": proposal.get("proposal_root_token"),
                    "proposal_type": proposal.get("proposal_type"),
                }

        revision_events = self.revision_controller(anchors, viability_out, arbiter_out)
        anchors = self.anchor_memory.apply_revision(anchors, revision_events)
        active_anchors = self.anchor_memory.get_active_anchors(anchors)
        diagnostics = self.anchor_memory.export_diagnostics(anchors)
        diagnostics["revision_event_count"] = len(revision_events)
        decision_map = {decision.anchor_id: decision for decision in revision_events}
        proposal_timing = self._proposal_timing_diagnostics(alternative_proposals, decision_map)

        anchor_context, proposal_diagnostics = self._build_anchor_context(
            hidden,
            detector_out["scores"],
            active_anchors,
            proposal_map=alternative_proposals,
            arbiter_out=arbiter_out,
            decision_map=decision_map,
        )
        proposal_diagnostics.update(proposal_timing)
        anchor_gate_delta = self.anchor_gate(anchor_context)
        conditioned_hidden = hidden + anchor_gate_delta
        logits = self.lm_head(conditioned_hidden)

        out = {
            "logits": logits,
            "hidden": hidden,
            "anchor_candidates": detector_out["candidates"],
            "active_anchors": active_anchors,
            "anchor_states": [[anchor.state for anchor in batch] for batch in anchors],
            "contradiction_pressure": contradiction_out["contradiction_pressure"],
            "viability": viability_out["viability"],
            "revision_events": revision_events,
            "anchor_diagnostics": diagnostics,
            "detector_scores": detector_out["scores"],
            "alternative_proposals": alternative_proposals,
            "proposal_diagnostics": proposal_diagnostics,
            "anchor_gate_delta": anchor_gate_delta,
        }

        if targets is not None:
            B, T, V = logits.shape
            ce_loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            detector_alignment_loss = F.mse_loss(
                detector_out["prior_scores"],
                detector_out["runtime_scores"],
            )
            context_stability_loss = anchor_gate_delta.pow(2).mean()
            total_loss = (
                ce_loss
                + self.cfg.anchor_detector_alignment_weight * detector_alignment_loss
                + self.cfg.anchor_context_stability_weight * context_stability_loss
            )
            out["loss"] = total_loss
            out["ce_loss"] = ce_loss
            out["component_losses"] = {
                "ce_loss": ce_loss,
                "detector_alignment_loss": detector_alignment_loss,
                "context_stability_loss": context_stability_loss,
            }

        return out

    def _build_anchor_context(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        active_anchors: list[list] | None = None,
        proposal_map: dict[int, dict] | None = None,
        arbiter_out: dict[int, dict] | None = None,
        decision_map: dict[int, object] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if active_anchors is None:
            weights = scores
        else:
            weights = torch.zeros_like(scores)
            for batch_idx, batch_anchors in enumerate(active_anchors):
                for anchor in batch_anchors:
                    start = max(0, min(int(anchor.start_idx), scores.size(1) - 1))
                    end = max(start, min(int(anchor.end_idx), scores.size(1) - 1))
                    weights[batch_idx, start:end + 1] = scores[batch_idx, start:end + 1]
        weights = weights.unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1, keepdim=True)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = pooled / denom

        base_diagnostics = {
            "proposal_count": len(proposal_map or {}),
            "regime_shift_count": sum(
                1 for proposal in (proposal_map or {}).values()
                if proposal.get("proposal_type") == "regime_shift_window"
            ),
            "anchors_with_proposal_influence": 0,
            "mean_proposal_score": 0.0,
            "mean_blend_ratio": 0.0,
        }
        if not proposal_map or active_anchors is None:
            return pooled.expand_as(hidden), base_diagnostics

        proposal_sum = torch.zeros_like(pooled)
        proposal_weight = torch.zeros(hidden.size(0), 1, 1, device=hidden.device, dtype=hidden.dtype)
        arbiter_out = arbiter_out or {}
        decision_map = decision_map or {}
        influenced = 0
        score_total = 0.0
        for batch_idx, batch_anchors in enumerate(active_anchors):
            for anchor in batch_anchors:
                proposal = proposal_map.get(anchor.id)
                if proposal is None or proposal.get("proposal_type") == "start_state_baseline":
                    continue
                arbiter = arbiter_out.get(anchor.id, {})
                decision = decision_map.get(anchor.id)
                alt_prob = float(arbiter.get("prefer_alt_prob", 0.0))
                if decision is None or getattr(decision, "action_probs", None) is None:
                    revise_prob = 0.0
                else:
                    revise_prob = float(decision.action_probs.get("revise", 0.0))
                proposal_strength = float(proposal.get("proposal_score", 0.0))
                gate = self._calibrate_proposal_gate(
                    proposal=proposal,
                    alt_prob=alt_prob,
                    revise_prob=revise_prob,
                )
                if gate <= 0.01:
                    continue
                proposal_sum[batch_idx, 0] += gate * proposal["repr"]
                proposal_weight[batch_idx, 0, 0] += gate
                influenced += 1
                score_total += proposal_strength

        proposal_mix = proposal_sum / proposal_weight.clamp_min(1e-6)
        blend_ratio = proposal_weight.clamp(max=0.25)
        final_context = (1.0 - blend_ratio) * pooled + blend_ratio * proposal_mix
        diagnostics = {
            "proposal_count": base_diagnostics["proposal_count"],
            "regime_shift_count": base_diagnostics["regime_shift_count"],
            "anchors_with_proposal_influence": influenced,
            "mean_proposal_score": score_total / max(influenced, 1),
            "mean_blend_ratio": float(blend_ratio.mean().item()),
        }
        return final_context.expand_as(hidden), diagnostics

    def _propose_alternative_reading(
        self,
        seq_hidden: torch.Tensor,
        seq_ids: torch.Tensor | None,
        anchor,
    ) -> dict:
        baseline = {
            "repr": seq_hidden[max(int(anchor.start_idx), 0)],
            "proposal_type": "start_state_baseline",
        }
        if seq_ids is None:
            return baseline

        span_start = max(int(anchor.start_idx), 0)
        span_end = min(int(anchor.end_idx), seq_ids.size(0) - 1)
        span_len = max(span_end - span_start + 1, 1)
        root_token = self._resolve_anchor_regime_root(seq_ids, span_start, span_end)
        if root_token is None:
            return baseline
        allowed_tokens = self.contradiction_monitor._REGIME_COMPATIBILITY.get(root_token)
        if allowed_tokens is None:
            return baseline

        start = min(span_end + 1, seq_ids.size(0))
        horizon = max(int(float(anchor.ttl) * 4), span_len * 4)
        stop = min(start + horizon, seq_ids.size(0))
        if stop - start < span_len:
            return baseline

        best_score = 0.0
        best_window: tuple[int, int] | None = None
        best_root_token: int | None = None
        for offset in range(start, stop - span_len + 1):
            window_tokens = seq_ids[offset: offset + span_len]
            regime_compatibility = sum(int(int(token.item()) in allowed_tokens) for token in window_tokens) / max(span_len, 1)
            conflict_strength = 1.0 - regime_compatibility
            if conflict_strength < 0.45:
                continue

            unique_tokens, counts = torch.unique(window_tokens, return_counts=True)
            dominant_ratio = float(counts.max().item()) / max(span_len, 1)
            incompatible_mask = torch.tensor(
                [int(token.item()) not in allowed_tokens for token in unique_tokens],
                device=unique_tokens.device,
                dtype=torch.bool,
            )
            if incompatible_mask.any():
                incompatible_tokens = unique_tokens[incompatible_mask]
                incompatible_counts = counts[incompatible_mask]
                candidate_root = int(incompatible_tokens[torch.argmax(incompatible_counts)].item())
                candidate_root_ratio = float(incompatible_counts.max().item()) / max(span_len, 1)
            else:
                candidate_root = int(window_tokens[0].item())
                candidate_root_ratio = dominant_ratio
            root_consistency = float((window_tokens == candidate_root).float().mean().item())

            future_tail = seq_ids[offset:stop]
            future_support = float((future_tail == candidate_root).float().mean().item())
            root_replacement_plausibility = 0.55 * candidate_root_ratio + 0.45 * future_support
            leading_shift_bonus = 0.0
            first_token = int(window_tokens[0].item())
            if first_token not in allowed_tokens:
                leading_shift_bonus = min(1.0, 0.5 + float((future_tail == first_token).float().mean().item()))
            coherence = (
                0.25 * dominant_ratio
                + 0.15 * root_consistency
                + 0.20 * future_support
                + 0.25 * root_replacement_plausibility
                + 0.15 * leading_shift_bonus
            )
            if coherence < 0.50:
                continue

            score = conflict_strength * coherence
            score *= self._proposal_root_prior(candidate_root)
            score *= self._anchor_local_proposal_prior(anchor, candidate_root)
            if score > best_score:
                best_score = score
                best_window = (offset, offset + span_len)
                best_root_token = candidate_root

        if best_window is None or best_score < 0.24:
            return baseline

        alt_start, alt_end = best_window
        alt_repr = seq_hidden[alt_start:alt_end].mean(dim=0)
        blend = min(0.35, best_score * 0.35)
        blended_repr = (1.0 - blend) * anchor.repr + blend * alt_repr
        return {
            "repr": blended_repr,
            "proposal_type": "regime_shift_window",
            "proposal_score": best_score,
            "proposal_span": (alt_start, alt_end - 1),
            "proposal_root_token": best_root_token,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _proposal_timing_diagnostics(
        proposal_map: dict[int, dict],
        decision_map: dict[int, object],
    ) -> dict[str, int]:
        revise_count = 0
        retire_count = 0
        strong_retire_count = 0
        strong_retire_gap_sum = 0.0
        for anchor_id, proposal in proposal_map.items():
            if proposal.get("proposal_type") == "start_state_baseline":
                continue
            decision = decision_map.get(anchor_id)
            if decision is None:
                continue
            if getattr(decision, "action", None) == "revise":
                revise_count += 1
            elif getattr(decision, "action", None) == "retire":
                retire_count += 1
                if float(proposal.get("proposal_score", 0.0)) >= 0.5:
                    strong_retire_count += 1
                    probs = getattr(decision, "action_probs", None) or {}
                    strong_retire_gap_sum += float(probs.get("retire", 0.0)) - float(probs.get("revise", 0.0))
        return {
            "proposal_revise_count": revise_count,
            "proposal_retire_count": retire_count,
            "strong_proposal_retire_count": strong_retire_count,
            "mean_strong_retire_gap": strong_retire_gap_sum / max(strong_retire_count, 1),
        }

    @classmethod
    def _calibrate_proposal_gate(
        cls,
        proposal: dict,
        alt_prob: float,
        revise_prob: float,
    ) -> float:
        proposal_strength = float(proposal.get("proposal_score", 0.0))
        root_token = proposal.get("proposal_root_token")
        multiplier = cls._PROPOSAL_GATE_MULTIPLIER.get(int(root_token), 1.0) if root_token is not None else 1.0
        return alt_prob * revise_prob * proposal_strength * multiplier

    @classmethod
    def _resolve_anchor_regime_root(
        cls,
        seq_ids: torch.Tensor,
        span_start: int,
        span_end: int,
    ) -> int | None:
        span_tokens = seq_ids[span_start: span_end + 1].tolist()
        for token in span_tokens:
            root = cls._ANCHOR_REGIME_ALIAS.get(int(token))
            if root is not None:
                return root
        return None

    @classmethod
    def _proposal_root_prior(cls, root_token: int | None) -> float:
        if root_token is None:
            return 1.0
        return cls._PROPOSAL_ROOT_PRIOR.get(int(root_token), 1.0)

    @staticmethod
    def _anchor_local_proposal_prior(anchor, root_token: int | None) -> float:
        if root_token != 45:
            return 1.0
        if float(anchor.contradiction_pressure) < 0.85:
            return 1.0
        descendant_coherence = float(anchor.descendant_coherence or 0.0)
        if descendant_coherence > 0.05:
            return 1.0
        return 0.45
