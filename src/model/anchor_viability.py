from __future__ import annotations

import math

import torch.nn as nn

from src.model.anchor_types import AnchorRecord, AnchorState
from src.model.config import ModelConfig


class ViabilityTracker(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        anchors: list[list[AnchorRecord]],
        contradiction: dict,
    ) -> dict:
        pressure_map = contradiction["contradiction_pressure"]
        viability: dict[int, float] = {}
        state_updates: dict[int, AnchorState] = {}

        for batch_anchors in anchors:
            for anchor in batch_anchors:
                support = self._to_float(anchor.support)
                pressure = float(pressure_map.get(anchor.id, self._to_float(anchor.contradiction_pressure)))
                age_penalty = 1.0 / max(self._to_float(anchor.ttl) + 1.0, 1.0)
                descendant_mass = self._to_float(anchor.descendant_mass or 0.0)
                descendant_coherence = self._to_float(anchor.descendant_coherence or 0.0)
                raw = (
                    self.cfg.anchor_viability_alpha * support
                    - self.cfg.anchor_viability_beta * pressure
                    - self.cfg.anchor_age_gamma * age_penalty
                    + self.cfg.anchor_descendant_mass_delta * descendant_mass
                    + self.cfg.anchor_descendant_coherence_eta * descendant_coherence
                )
                current_viability = 1.0 / (1.0 + math.exp(-raw))
                anchor.viability = current_viability
                viability[anchor.id] = current_viability

                if anchor.state == AnchorState.CANDIDATE:
                    if current_viability >= self.cfg.anchor_confirm_threshold and pressure <= self.cfg.anchor_contradiction_threshold:
                        next_state = AnchorState.CONFIRMED
                    elif current_viability >= self.cfg.anchor_candidate_promote_threshold:
                        next_state = AnchorState.PROVISIONAL
                    elif pressure >= self.cfg.anchor_dead_end_threshold and self._to_float(anchor.ttl) <= 1.0:
                        next_state = AnchorState.DEAD_END
                    else:
                        next_state = AnchorState.CANDIDATE
                elif anchor.state == AnchorState.PROVISIONAL:
                    if current_viability >= self.cfg.anchor_confirm_threshold and pressure <= self.cfg.anchor_contradiction_threshold:
                        next_state = AnchorState.CONFIRMED
                    elif current_viability <= self.cfg.anchor_revision_threshold and pressure >= self.cfg.anchor_contradiction_threshold:
                        next_state = AnchorState.DEAD_END
                    else:
                        next_state = AnchorState.PROVISIONAL
                elif anchor.state == AnchorState.CONFIRMED:
                    if self._to_float(anchor.ttl) <= 1.0:
                        next_state = AnchorState.DECAYING
                    elif current_viability <= self.cfg.anchor_revision_threshold and pressure >= self.cfg.anchor_contradiction_threshold:
                        next_state = AnchorState.DEAD_END
                    else:
                        next_state = AnchorState.CONFIRMED
                elif anchor.state == AnchorState.DECAYING:
                    if current_viability <= self.cfg.anchor_revision_threshold:
                        next_state = AnchorState.DEAD_END
                    else:
                        next_state = AnchorState.DECAYING
                else:
                    next_state = AnchorState.DEAD_END

                state_updates[anchor.id] = next_state
        return {
            "viability": viability,
            "state_updates": state_updates,
        }

    @staticmethod
    def _to_float(value: float) -> float:
        return float(value)
