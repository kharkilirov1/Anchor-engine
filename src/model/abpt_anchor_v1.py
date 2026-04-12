from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.anchor_detector import AnchorDetector
from src.model.anchor_memory import AnchorMemory
from src.model.anchor_monitor import ContradictionMonitor
from src.model.anchor_revision import AnchorArbiter, RevisionController
from src.model.anchor_types import AnchorState
from src.model.anchor_viability import ViabilityTracker
from src.model.backbone import Backbone
from src.model.config import ModelConfig
from src.model.fog_flow import FogFlowBackbone
from src.model.future_proposal import FutureProposalHead
from src.model.proposal_rollout import ProposalRolloutBranch


class ABPTAnchorV1(nn.Module):
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
        self.backbone = FogFlowBackbone(cfg) if cfg.use_fog_flow else Backbone(cfg)
        self.anchor_detector = AnchorDetector(cfg)
        self.anchor_memory = AnchorMemory(cfg)
        self.contradiction_monitor = ContradictionMonitor(cfg)
        self.viability_tracker = ViabilityTracker(cfg)
        self.anchor_arbiter = AnchorArbiter(cfg)
        self.revision_controller = RevisionController(cfg)
        self.future_proposal_head = FutureProposalHead(cfg) if cfg.anchor_use_future_proposal_head else None
        self.proposal_rollout = ProposalRolloutBranch(cfg) if cfg.anchor_use_proposal_rollout else None
        self.anchor_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
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
        proposal_domain_mode = self._proposal_domain_mode()

        arbiter_out: dict[int, dict] = {}
        alternative_proposals: dict[int, dict] = {}
        for batch_idx, batch_anchors in enumerate(anchors):
            seq_hidden = hidden[batch_idx]
            seq_ids = input_ids[batch_idx]
            for anchor in batch_anchors:
                pressure = contradiction_out["contradiction_pressure"].get(anchor.id, 0.0)
                if not self._should_request_proposal(
                    pressure=pressure,
                    domain_mode=proposal_domain_mode,
                ):
                    continue
                proposal = self._propose_alternative_reading(
                    seq_hidden=seq_hidden,
                    seq_ids=seq_ids,
                    anchor=anchor,
                )
                proposal = self._attach_proposal_rollout(
                    seq_hidden=seq_hidden,
                    anchor=anchor,
                    proposal=proposal,
                )
                alternative_proposals[anchor.id] = proposal
                arbiter_result = self.anchor_arbiter(
                    seq_hidden,
                    anchor,
                    alt={"repr": self._proposal_effective_repr(proposal)},
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

        anchor_context, proposal_diagnostics, anchor_gate_strength = self._build_anchor_context(
            hidden,
            detector_out["scores"],
            active_anchors,
            proposal_map=alternative_proposals,
            arbiter_out=arbiter_out,
            decision_map=decision_map,
        )
        proposal_diagnostics.update(proposal_timing)
        anchor_gate_delta = self.anchor_gate(anchor_context)
        conditioned_hidden = hidden + anchor_gate_strength * anchor_gate_delta
        logits = self.lm_head(conditioned_hidden)
        proposal_aux_losses = self._proposal_aux_losses(
            hidden=hidden,
            anchors=anchors,
            proposal_map=alternative_proposals,
            targets=targets,
        )

        out = {
            "logits": logits,
            "hidden": hidden,
            "backbone_hidden": hidden,
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
            "anchor_gate_strength": anchor_gate_strength,
            "flow_type": backbone_out.get("flow_type", "transformer"),
            "fog_profile": backbone_out.get("fog_profile"),
            "fog_layers": backbone_out.get("fog_layers", []),
        }

        if targets is not None:
            B, T, V = logits.shape
            ce_loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            detector_alignment_loss = F.mse_loss(
                detector_out["prior_scores"],
                detector_out["runtime_scores"],
            )
            context_stability_loss = anchor_gate_delta.pow(2).mean()
            proposal_score_loss = proposal_aux_losses["proposal_score_loss"]
            proposal_margin_loss = proposal_aux_losses["proposal_margin_loss"]
            proposal_alignment_loss = proposal_aux_losses["proposal_alignment_loss"]
            proposal_counterfactual_loss = proposal_aux_losses["proposal_counterfactual_loss"]
            proposal_rollout_loss = proposal_aux_losses["proposal_rollout_loss"]
            total_loss = (
                ce_loss
                + self.cfg.anchor_detector_alignment_weight * detector_alignment_loss
                + self.cfg.anchor_context_stability_weight * context_stability_loss
                + self.cfg.anchor_proposal_score_weight * proposal_score_loss
                + self.cfg.anchor_proposal_margin_weight * proposal_margin_loss
                + self.cfg.anchor_proposal_alignment_weight * proposal_alignment_loss
                + self.cfg.anchor_proposal_counterfactual_weight * proposal_counterfactual_loss
                + self.cfg.anchor_proposal_rollout_weight * proposal_rollout_loss
            )
            out["loss"] = total_loss
            out["ce_loss"] = ce_loss
            out["component_losses"] = {
                "ce_loss": ce_loss,
                "detector_alignment_loss": detector_alignment_loss,
                "context_stability_loss": context_stability_loss,
                "proposal_score_loss": proposal_score_loss,
                "proposal_margin_loss": proposal_margin_loss,
                "proposal_alignment_loss": proposal_alignment_loss,
                "proposal_counterfactual_loss": proposal_counterfactual_loss,
                "proposal_rollout_loss": proposal_rollout_loss,
            }
            out["proposal_aux_metrics"] = {
                "proposal_counterfactual_gain": proposal_aux_losses["proposal_counterfactual_gain"],
                "proposal_counterfactual_current_ce": proposal_aux_losses["proposal_counterfactual_current_ce"],
                "proposal_counterfactual_proposal_ce": proposal_aux_losses["proposal_counterfactual_proposal_ce"],
                "proposal_counterfactual_count": proposal_aux_losses["proposal_counterfactual_count"],
                "proposal_rollout_gain": proposal_aux_losses["proposal_rollout_gain"],
                "proposal_rollout_count": proposal_aux_losses["proposal_rollout_count"],
                "proposal_rollout_depth": proposal_aux_losses["proposal_rollout_depth"],
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
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        state_scale = {
            AnchorState.CANDIDATE: 0.25,
            AnchorState.PROVISIONAL: 0.75,
            AnchorState.CONFIRMED: 1.0,
            AnchorState.DECAYING: 0.5,
        }
        scores = scores.detach()
        if active_anchors is None:
            weights = scores
        else:
            weights = torch.zeros_like(scores)
            for batch_idx, batch_anchors in enumerate(active_anchors):
                for anchor in batch_anchors:
                    viability = float(anchor.viability)
                    if anchor.state == AnchorState.CANDIDATE and viability < self.cfg.anchor_context_min_viability:
                        continue
                    start = max(0, min(int(anchor.start_idx), scores.size(1) - 1))
                    end = max(start, min(int(anchor.end_idx), scores.size(1) - 1))
                    anchor_weight = (
                        scores[batch_idx, start:end + 1]
                        * max(viability, 0.05)
                        * state_scale.get(anchor.state, 0.0)
                    )
                    weights[batch_idx, start:end + 1] = torch.maximum(
                        weights[batch_idx, start:end + 1],
                        anchor_weight,
                    )
        weights = weights.unsqueeze(-1)
        if float(weights.sum().item()) <= 1e-6:
            empty_context = torch.zeros_like(hidden)
            diagnostics = {
                "proposal_count": len(proposal_map or {}),
                "non_baseline_proposal_count": 0,
                "regime_shift_count": 0,
                "future_window_count": 0,
                "rollout_count": 0,
                "mean_rollout_steps": 0.0,
                "anchors_with_proposal_influence": 0,
                "mean_proposal_score": 0.0,
                "mean_blend_ratio": 0.0,
                "context_anchor_mass": 0.0,
            }
            gate_strength = torch.zeros(hidden.size(0), hidden.size(1), 1, device=hidden.device, dtype=hidden.dtype)
            return empty_context, diagnostics, gate_strength
        pooled = (hidden * weights).sum(dim=1, keepdim=True)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = pooled / denom
        gate_strength = weights / weights.amax(dim=1, keepdim=True).clamp_min(1e-6)

        base_diagnostics = {
            "proposal_count": len(proposal_map or {}),
            "non_baseline_proposal_count": sum(
                1 for proposal in (proposal_map or {}).values()
                if proposal.get("proposal_type") != "start_state_baseline"
            ),
            "regime_shift_count": sum(
                1 for proposal in (proposal_map or {}).values()
                if proposal.get("proposal_type") == "regime_shift_window"
            ),
            "future_window_count": sum(
                1 for proposal in (proposal_map or {}).values()
                if proposal.get("proposal_type") == "future_window_head"
            ),
            "rollout_count": sum(
                1 for proposal in (proposal_map or {}).values()
                if int(proposal.get("rollout_steps", 0)) > 0
            ),
            "mean_rollout_steps": (
                sum(float(proposal.get("rollout_steps", 0)) for proposal in (proposal_map or {}).values())
                / max(len(proposal_map or {}), 1)
            ),
            "anchors_with_proposal_influence": 0,
            "mean_proposal_score": 0.0,
            "mean_blend_ratio": 0.0,
            "context_anchor_mass": float(weights.sum(dim=1).mean().item()),
        }
        if not proposal_map or active_anchors is None:
            return pooled.expand_as(hidden), base_diagnostics, gate_strength

        proposal_sum = torch.zeros_like(pooled)
        proposal_weight = torch.zeros(hidden.size(0), 1, 1, device=hidden.device, dtype=hidden.dtype)
        arbiter_out = arbiter_out or {}
        decision_map = decision_map or {}
        influenced = 0
        score_total = 0.0
        rollout_steps_total = 0.0
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
                proposal_sum[batch_idx, 0] += gate * self._proposal_effective_repr(proposal)
                proposal_weight[batch_idx, 0, 0] += gate
                influenced += 1
                score_total += proposal_strength
                rollout_steps_total += float(proposal.get("rollout_steps", 0))

        proposal_mix = proposal_sum / proposal_weight.clamp_min(1e-6)
        blend_ratio = proposal_weight.clamp(max=0.25)
        final_context = (1.0 - blend_ratio) * pooled + blend_ratio * proposal_mix
        diagnostics = {
            "proposal_count": base_diagnostics["proposal_count"],
            "non_baseline_proposal_count": base_diagnostics["non_baseline_proposal_count"],
            "regime_shift_count": base_diagnostics["regime_shift_count"],
            "future_window_count": base_diagnostics["future_window_count"],
            "rollout_count": base_diagnostics["rollout_count"],
            "mean_rollout_steps": rollout_steps_total / max(influenced, 1),
            "anchors_with_proposal_influence": influenced,
            "mean_proposal_score": score_total / max(influenced, 1),
            "mean_blend_ratio": float(blend_ratio.mean().item()),
            "context_anchor_mass": base_diagnostics["context_anchor_mass"],
        }
        return final_context.expand_as(hidden), diagnostics, gate_strength

    @staticmethod
    def _cosine01_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).mean()
        return (cosine + 1.0) * 0.5

    @staticmethod
    def _proposal_effective_repr(proposal: dict) -> torch.Tensor:
        branch_repr = proposal.get("branch_repr")
        if isinstance(branch_repr, torch.Tensor):
            return branch_repr
        return proposal["repr"]

    def _attach_proposal_rollout(
        self,
        seq_hidden: torch.Tensor,
        anchor,
        proposal: dict,
    ) -> dict:
        if proposal.get("proposal_type") == "start_state_baseline":
            return proposal
        if not self.cfg.anchor_use_proposal_rollout or self.proposal_rollout is None:
            return proposal
        if self._proposal_domain_mode() == "synthetic":
            return proposal
        if float(anchor.contradiction_pressure) < float(self.cfg.anchor_proposal_rollout_pressure_trigger):
            return proposal
        if float(proposal.get("proposal_score", 0.0)) < float(self.cfg.anchor_proposal_rollout_score_trigger):
            return proposal
        context_idx = max(0, min(int(anchor.end_idx), seq_hidden.size(0) - 1))
        rollout = self.proposal_rollout(
            anchor_repr=anchor.repr,
            proposal_repr=proposal["repr"],
            context_repr=seq_hidden[context_idx],
        )
        branch_mix = min(
            0.5,
            float(self.cfg.anchor_proposal_rollout_residual_scale) * max(float(proposal.get("proposal_score", 0.0)), 0.25),
        )
        branch_repr = (1.0 - branch_mix) * proposal["repr"] + branch_mix * rollout["summary"]
        return {
            **proposal,
            "branch_repr": branch_repr,
            "rollout_summary": rollout["summary"],
            "rollout_states": rollout["states"],
            "rollout_steps": int(rollout["states"].size(0)),
            "rollout_mix": branch_mix,
        }

    def _proposal_aux_losses(
        self,
        hidden: torch.Tensor,
        anchors: list[list],
        proposal_map: dict[int, dict],
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = hidden.sum() * 0.0
        if not proposal_map:
            return {
                "proposal_score_loss": zero,
                "proposal_margin_loss": zero,
                "proposal_alignment_loss": zero,
                "proposal_counterfactual_loss": zero,
                "proposal_rollout_loss": zero,
                "proposal_counterfactual_gain": zero,
                "proposal_rollout_gain": zero,
                "proposal_counterfactual_current_ce": zero,
                "proposal_counterfactual_proposal_ce": zero,
                "proposal_counterfactual_count": zero,
                "proposal_rollout_count": zero,
                "proposal_rollout_depth": zero,
            }

        score_terms: list[torch.Tensor] = []
        margin_terms: list[torch.Tensor] = []
        alignment_terms: list[torch.Tensor] = []
        counterfactual_terms: list[torch.Tensor] = []
        rollout_terms: list[torch.Tensor] = []
        counterfactual_gain_terms: list[torch.Tensor] = []
        rollout_gain_terms: list[torch.Tensor] = []
        counterfactual_current_ce_terms: list[torch.Tensor] = []
        counterfactual_proposal_ce_terms: list[torch.Tensor] = []
        counterfactual_count_terms: list[torch.Tensor] = []
        rollout_count_terms: list[torch.Tensor] = []
        rollout_depth_terms: list[torch.Tensor] = []
        margin_target = hidden.new_tensor(float(self.cfg.anchor_proposal_margin_target))
        target_temp = max(float(self.cfg.anchor_proposal_target_temperature), 1e-6)
        trigger = float(self.cfg.anchor_future_proposal_trigger)
        counterfactual_margin = hidden.new_tensor(float(self.cfg.anchor_proposal_counterfactual_margin))
        rollout_margin = hidden.new_tensor(float(self.cfg.anchor_proposal_rollout_margin))

        for batch_idx, batch_anchors in enumerate(anchors):
            seq_hidden = hidden[batch_idx]
            seq_len = seq_hidden.size(0)
            seq_targets = targets[batch_idx] if targets is not None else None
            for anchor in batch_anchors:
                proposal = proposal_map.get(anchor.id)
                if proposal is None or proposal.get("proposal_type") == "start_state_baseline":
                    continue
                proposal_span = proposal.get("proposal_span")
                if proposal_span is None:
                    continue
                proposal_start = max(0, min(int(proposal_span[0]), seq_len - 1))
                proposal_end = max(proposal_start, min(int(proposal_span[1]), seq_len - 1))
                future_start = min(proposal_end + 1, seq_len)
                horizon = max(
                    int(float(anchor.ttl) * float(self.cfg.anchor_future_proposal_horizon_scale)),
                    max(1, proposal_end - proposal_start + 1) * 2,
                )
                future_stop = min(seq_len, future_start + max(horizon, 1))
                if future_start < future_stop:
                    future_target = seq_hidden[future_start:future_stop].mean(dim=0)
                else:
                    future_target = seq_hidden[proposal_start:proposal_end + 1].mean(dim=0)

                proposal_repr = self._proposal_effective_repr(proposal)
                current_sim = self._cosine01_tensor(anchor.repr, future_target)
                proposal_sim = self._cosine01_tensor(proposal_repr, future_target)
                gain = proposal_sim - current_sim

                pressure = float(anchor.contradiction_pressure)
                viability = float(anchor.viability)
                pressure_gap = max(0.0, pressure - trigger)
                if pressure_gap <= 0.0:
                    continue
                weight = (pressure_gap / max(1e-6, 1.0 - trigger)) * max(0.1, 1.0 - viability)
                weight_tensor = hidden.new_tensor(weight)

                score_tensor = proposal.get("proposal_score_tensor")
                if isinstance(score_tensor, torch.Tensor):
                    score_target = torch.sigmoid(
                        (gain.detach() + hidden.new_tensor(0.5 * pressure_gap)) / target_temp
                    ).clamp(1e-5, 1.0 - 1e-5)
                    score_terms.append(
                        weight_tensor * F.binary_cross_entropy(score_tensor.clamp(1e-5, 1.0 - 1e-5), score_target)
                    )

                margin_terms.append(weight_tensor * F.relu(margin_target - gain))
                alignment_terms.append(weight_tensor * (1.0 - proposal_sim))

                counterfactual_targets = self._proposal_counterfactual_targets(
                    seq_targets=seq_targets,
                    proposal_start=proposal_start,
                    proposal_end=proposal_end,
                )
                if counterfactual_targets is None:
                    continue

                proposal_logits = self.lm_head(proposal_repr).unsqueeze(0).expand(counterfactual_targets.size(0), -1)
                current_logits = self.lm_head(anchor.repr.detach()).unsqueeze(0).expand(counterfactual_targets.size(0), -1)
                proposal_ce = F.cross_entropy(proposal_logits, counterfactual_targets, reduction="mean")
                current_ce = F.cross_entropy(current_logits, counterfactual_targets, reduction="mean")
                current_ce_detached = current_ce.detach()
                counterfactual_terms.append(
                    weight_tensor * F.relu(proposal_ce - current_ce_detached + counterfactual_margin)
                )
                counterfactual_gain_terms.append(weight_tensor * (current_ce_detached - proposal_ce.detach()))
                counterfactual_current_ce_terms.append(weight_tensor * current_ce_detached)
                counterfactual_proposal_ce_terms.append(weight_tensor * proposal_ce.detach())
                counterfactual_count_terms.append(weight_tensor.new_tensor(1.0))

                rollout_states = proposal.get("rollout_states")
                if isinstance(rollout_states, torch.Tensor) and rollout_states.numel() > 0:
                    rollout_steps = min(counterfactual_targets.size(0), rollout_states.size(0))
                    rollout_targets = counterfactual_targets[:rollout_steps]
                    rollout_logits = self.lm_head(rollout_states[:rollout_steps])
                    rollout_ce = F.cross_entropy(rollout_logits, rollout_targets, reduction="mean")
                    rollout_current_logits = self.lm_head(anchor.repr.detach()).unsqueeze(0).expand(rollout_steps, -1)
                    rollout_current_ce = F.cross_entropy(rollout_current_logits, rollout_targets, reduction="mean")
                    rollout_current_ce_detached = rollout_current_ce.detach()
                    rollout_terms.append(
                        weight_tensor * F.relu(rollout_ce - rollout_current_ce_detached + rollout_margin)
                    )
                    rollout_gain_terms.append(weight_tensor * (rollout_current_ce_detached - rollout_ce.detach()))
                    rollout_count_terms.append(weight_tensor.new_tensor(1.0))
                    rollout_depth_terms.append(weight_tensor.new_tensor(float(rollout_steps)))

        return {
            "proposal_score_loss": torch.stack(score_terms).mean() if score_terms else zero,
            "proposal_margin_loss": torch.stack(margin_terms).mean() if margin_terms else zero,
            "proposal_alignment_loss": torch.stack(alignment_terms).mean() if alignment_terms else zero,
            "proposal_counterfactual_loss": torch.stack(counterfactual_terms).mean() if counterfactual_terms else zero,
            "proposal_rollout_loss": torch.stack(rollout_terms).mean() if rollout_terms else zero,
            "proposal_counterfactual_gain": torch.stack(counterfactual_gain_terms).mean() if counterfactual_gain_terms else zero,
            "proposal_rollout_gain": torch.stack(rollout_gain_terms).mean() if rollout_gain_terms else zero,
            "proposal_counterfactual_current_ce": torch.stack(counterfactual_current_ce_terms).mean() if counterfactual_current_ce_terms else zero,
            "proposal_counterfactual_proposal_ce": torch.stack(counterfactual_proposal_ce_terms).mean() if counterfactual_proposal_ce_terms else zero,
            "proposal_counterfactual_count": torch.stack(counterfactual_count_terms).sum() if counterfactual_count_terms else zero,
            "proposal_rollout_count": torch.stack(rollout_count_terms).sum() if rollout_count_terms else zero,
            "proposal_rollout_depth": torch.stack(rollout_depth_terms).mean() if rollout_depth_terms else zero,
        }

    def _proposal_counterfactual_targets(
        self,
        seq_targets: torch.Tensor | None,
        proposal_start: int,
        proposal_end: int,
    ) -> torch.Tensor | None:
        if seq_targets is None or seq_targets.numel() == 0:
            return None
        seq_len = seq_targets.size(0)
        tail = max(1, int(self.cfg.anchor_proposal_counterfactual_window))
        target_start = max(0, min(proposal_start, seq_len - 1))
        target_stop = min(seq_len, proposal_end + 1 + tail)
        if target_start >= target_stop:
            future_start = min(seq_len, proposal_end + 1)
            future_stop = min(seq_len, future_start + tail)
            if future_start >= future_stop:
                return None
            target_start, target_stop = future_start, future_stop
        return seq_targets[target_start:target_stop]

    def _proposal_domain_mode(self) -> str:
        if self.cfg.anchor_domain_mode in {"synthetic", "real"}:
            return self.cfg.anchor_domain_mode
        return "real"

    def _should_request_proposal(
        self,
        pressure: float,
        domain_mode: str,
    ) -> bool:
        if domain_mode == "synthetic":
            threshold = float(self.cfg.anchor_dead_end_threshold)
        else:
            threshold = float(self.cfg.anchor_future_proposal_trigger)
        return float(pressure) >= threshold

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
        if self.cfg.anchor_domain_mode == "synthetic":
            if seq_ids is None:
                return baseline
            return self._propose_synthetic_alternative_reading(
                seq_hidden=seq_hidden,
                seq_ids=seq_ids,
                anchor=anchor,
                baseline=baseline,
            )

        if not self.cfg.anchor_use_future_proposal_head or self.future_proposal_head is None:
            return baseline

        proposal = self.future_proposal_head.propose(
            seq_hidden=seq_hidden,
            seq_ids=seq_ids,
            anchor=anchor,
        )
        if proposal is None:
            return baseline
        return proposal

    def _propose_synthetic_alternative_reading(
        self,
        seq_hidden: torch.Tensor,
        seq_ids: torch.Tensor,
        anchor,
        baseline: dict,
    ) -> dict:

        span_start = max(int(anchor.start_idx), 0)
        span_end = min(int(anchor.end_idx), seq_ids.size(0) - 1)
        span_len = max(span_end - span_start + 1, 1)
        anchor_span = seq_ids[span_start: span_end + 1]
        root_token = self._resolve_anchor_regime_root(seq_ids, span_start, span_end)
        allowed_tokens = self.contradiction_monitor._REGIME_COMPATIBILITY.get(root_token)

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
            regime_compatibility = self.contradiction_monitor.regime_compatibility_score(
                window_tokens=window_tokens,
                anchor_span=anchor_span,
                root_token=root_token,
            )
            conflict_strength = 1.0 - regime_compatibility
            if conflict_strength < 0.45:
                continue

            unique_tokens, counts = torch.unique(window_tokens, return_counts=True)
            dominant_ratio = float(counts.max().item()) / max(span_len, 1)
            incompatible_mask = self._proposal_incompatible_mask(
                unique_tokens=unique_tokens,
                anchor_span=anchor_span,
                root_token=root_token,
                allowed_tokens=allowed_tokens,
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
            if allowed_tokens is None:
                if first_token not in {int(token) for token in anchor_span.tolist()}:
                    leading_shift_bonus = min(1.0, 0.5 + float((future_tail == first_token).float().mean().item()))
            elif first_token not in allowed_tokens:
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
        return ContradictionMonitor.infer_reference_root(seq_ids[span_start: span_end + 1])

    @classmethod
    def _proposal_root_prior(cls, root_token: int | None) -> float:
        if root_token is None:
            return 1.0
        return cls._PROPOSAL_ROOT_PRIOR.get(int(root_token), 1.0)

    @staticmethod
    def _proposal_incompatible_mask(
        unique_tokens: torch.Tensor,
        anchor_span: torch.Tensor,
        root_token: int | None,
        allowed_tokens: set[int] | None,
    ) -> torch.Tensor:
        anchor_token_set = {int(token) for token in anchor_span.tolist()}
        incompatible: list[bool] = []
        for token in unique_tokens.tolist():
            token = int(token)
            token_alias = ContradictionMonitor._REGIME_ROOT_ALIAS.get(token)
            if allowed_tokens is not None:
                incompatible.append(token not in allowed_tokens and token_alias != root_token)
            else:
                incompatible.append(token not in anchor_token_set and token_alias != root_token)
        return torch.tensor(incompatible, device=unique_tokens.device, dtype=torch.bool)

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
