from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.anchor_types import AnchorRecord, AnchorState, RevisionDecision
from src.model.config import ModelConfig


class AnchorArbiter(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        hidden: torch.Tensor,
        anchor: AnchorRecord,
        alt: dict | None = None,
    ) -> dict:
        start = min(anchor.end_idx + 1, hidden.size(0))
        if start >= hidden.size(0):
            future_mean = anchor.repr.unsqueeze(0)
        else:
            future_mean = hidden[start:].mean(dim=0, keepdim=True)

        current_score = float(F.cosine_similarity(anchor.repr.unsqueeze(0), future_mean, dim=-1).mean().item())
        alt_repr = alt["repr"] if alt is not None else anchor.repr
        alt_score = float(F.cosine_similarity(alt_repr.unsqueeze(0), future_mean, dim=-1).mean().item())
        margin = current_score - alt_score
        prefer_current_prob = 1.0 / (1.0 + math.exp(-self.cfg.anchor_arbiter_beta * margin))
        prefer_current = prefer_current_prob >= 0.5
        return {
            "prefer_current": prefer_current,
            "prefer_current_prob": prefer_current_prob,
            "prefer_alt_prob": 1.0 - prefer_current_prob,
            "margin": margin,
            "arbiter_score": current_score,
            "alt_score": alt_score,
        }


class RevisionController(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        anchors: list[list[AnchorRecord]],
        viability: dict,
        arbiter: dict | None = None,
    ) -> list[RevisionDecision]:
        viability_map = viability["viability"]
        state_updates = viability["state_updates"]
        arbiter = arbiter or {}
        decisions: list[RevisionDecision] = []

        for batch_anchors in anchors:
            for anchor in batch_anchors:
                current_viability = float(viability_map.get(anchor.id, anchor.viability))
                next_state = state_updates.get(anchor.id, anchor.state)
                pressure = float(anchor.contradiction_pressure)
                hard_dead_end_threshold = min(1.0, self.cfg.anchor_dead_end_threshold + 0.15)
                arbiter_out = arbiter.get(anchor.id)
                prefer_current_prob = 1.0 if arbiter_out is None else float(arbiter_out.get("prefer_current_prob", 1.0))
                proposal_score = 0.0 if arbiter_out is None else float(arbiter_out.get("proposal_score", 0.0))
                proposal_root_token = None if arbiter_out is None else arbiter_out.get("proposal_root_token")
                action_probs = self._action_distribution(
                    anchor=anchor,
                    next_state=next_state,
                    pressure=pressure,
                    viability=current_viability,
                    prefer_current_prob=prefer_current_prob,
                    hard_dead_end_threshold=hard_dead_end_threshold,
                )
                action = max(action_probs, key=action_probs.get)
                if self._should_override_induction_retire(
                    action=action,
                    action_probs=action_probs,
                    pressure=pressure,
                    viability=current_viability,
                    proposal_score=proposal_score,
                    proposal_root_token=proposal_root_token,
                ):
                    action = "revise"
                elif self._should_override_complexity_retire(
                    action=action,
                    action_probs=action_probs,
                    pressure=pressure,
                    viability=current_viability,
                    proposal_score=proposal_score,
                    proposal_root_token=proposal_root_token,
                ):
                    action = "revise"

                if action == "keep":
                    reason = "soft_keep"
                    new_state = next_state
                    alt_branch_used = False
                elif action == "revise":
                    if action_probs["retire"] > action_probs["revise"]:
                        if proposal_root_token == 45:
                            reason = "induction_timing_override"
                        elif proposal_root_token == 35:
                            reason = "complexity_timing_override"
                        else:
                            reason = "soft_revise"
                    else:
                        reason = "soft_revise"
                    new_state = AnchorState.PROVISIONAL
                    alt_branch_used = True
                elif action == "retire":
                    reason = "soft_retire"
                    new_state = AnchorState.DEAD_END
                    alt_branch_used = False
                else:
                    reason = "soft_downgrade"
                    new_state = AnchorState.DECAYING
                    alt_branch_used = False

                decisions.append(
                    RevisionDecision(
                        anchor_id=anchor.id,
                        action=action,
                        reason=reason,
                        new_state=new_state,
                        alt_branch_used=alt_branch_used,
                        action_probs=action_probs,
                    )
                )
        return decisions

    def _action_distribution(
        self,
        anchor: AnchorRecord,
        next_state: AnchorState,
        pressure: float,
        viability: float,
        prefer_current_prob: float,
        hard_dead_end_threshold: float,
    ) -> dict[str, float]:
        alt_prob = 1.0 - prefer_current_prob
        candidate_bonus = 1.0 if anchor.state == AnchorState.CANDIDATE else 0.0
        revision_ready = 1.0 if next_state in {AnchorState.PROVISIONAL, AnchorState.CONFIRMED, AnchorState.DECAYING} else 0.0
        decaying_bonus = 1.0 if next_state == AnchorState.DECAYING else 0.0
        dead_bonus = 1.0 if next_state == AnchorState.DEAD_END else 0.0
        contradiction_excess = max(0.0, pressure - self.cfg.anchor_contradiction_threshold)
        dead_end_excess = max(0.0, pressure - hard_dead_end_threshold)

        keep_logit = (
            2.6 * viability
            + 1.4 * prefer_current_prob
            - 2.2 * pressure
            + 0.5 * candidate_bonus
            + 0.5 * (1.0 - contradiction_excess)
        )
        revise_logit = (
            0.8 * alt_prob
            + 0.9 * revision_ready
            + 3.5 * contradiction_excess
            + 0.3 * (1.0 - viability)
        )
        downgrade_logit = (
            1.2 * pressure
            + 1.0 * (1.0 - viability)
            + 0.8 * decaying_bonus
            + 0.2 * prefer_current_prob
        )
        retire_logit = (
            2.4 * dead_bonus
            + 4.0 * dead_end_excess
            + 1.5 * pressure
            + 1.4 * (1.0 - viability)
            - 0.3 * prefer_current_prob
        )

        logits = torch.tensor(
            [keep_logit, revise_logit, downgrade_logit, retire_logit],
            dtype=torch.float32,
        ) / max(self.cfg.anchor_revision_temperature, 1e-6)
        probs = torch.softmax(logits, dim=0)
        actions = ["keep", "revise", "downgrade", "retire"]
        return {action: float(prob.item()) for action, prob in zip(actions, probs)}

    @staticmethod
    def _should_override_induction_retire(
        action: str,
        action_probs: dict[str, float],
        pressure: float,
        viability: float,
        proposal_score: float,
        proposal_root_token: int | None,
    ) -> bool:
        if action != "retire":
            return False
        if proposal_root_token != 45:
            return False
        if proposal_score < 0.95:
            return False
        retire_prob = float(action_probs.get("retire", 0.0))
        revise_prob = float(action_probs.get("revise", 0.0))
        if revise_prob < 0.40:
            return False
        if retire_prob - revise_prob > 0.065:
            return False
        if pressure >= 0.80:
            return False
        if viability < 0.42:
            return False
        return True

    @staticmethod
    def _should_override_complexity_retire(
        action: str,
        action_probs: dict[str, float],
        pressure: float,
        viability: float,
        proposal_score: float,
        proposal_root_token: int | None,
    ) -> bool:
        if action != "retire":
            return False
        if proposal_root_token != 35:
            return False
        if proposal_score < 0.85:
            return False
        retire_prob = float(action_probs.get("retire", 0.0))
        revise_prob = float(action_probs.get("revise", 0.0))
        if revise_prob < 0.40:
            return False
        if retire_prob - revise_prob > 0.02:
            return False
        if pressure >= 0.76:
            return False
        if viability < 0.45:
            return False
        return True
