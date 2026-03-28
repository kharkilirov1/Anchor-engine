from __future__ import annotations

import torch
import torch.nn as nn

from src.model.anchor_types import AnchorCandidate, AnchorRecord, AnchorState, RevisionDecision
from src.model.config import ModelConfig


class AnchorMemory(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self._next_anchor_id = 0

    def add_candidates(
        self,
        candidates: list[list[AnchorCandidate]],
        anchors: list[list[AnchorRecord]] | None = None,
    ) -> list[list[AnchorRecord]]:
        if anchors is None:
            anchors = [[] for _ in candidates]

        for batch_anchors, batch_candidates in zip(anchors, candidates):
            for candidate in batch_candidates:
                batch_anchors.append(
                    AnchorRecord(
                        id=self._next_anchor_id,
                        start_idx=candidate.start_idx,
                        end_idx=candidate.end_idx,
                        repr=candidate.repr,
                        score=candidate.score,
                        state=AnchorState.CANDIDATE,
                        support=self._to_float(candidate.score),
                        contradiction_pressure=0.0,
                        viability=self._to_float(candidate.score),
                        ttl=float(self.cfg.anchor_ttl_init),
                        descendant_mass=0.0,
                        descendant_coherence=0.0,
                    )
                )
                self._next_anchor_id += 1
        return anchors

    def update_support(
        self,
        anchors: list[list[AnchorRecord]],
        detector_scores: torch.Tensor | None = None,
    ) -> list[list[AnchorRecord]]:
        for batch_idx, batch_anchors in enumerate(anchors):
            for anchor in batch_anchors:
                if detector_scores is not None and anchor.end_idx < detector_scores.size(1):
                    current = float(detector_scores[batch_idx, anchor.end_idx].item())
                else:
                    current = self._to_float(anchor.score)
                anchor.support = self.cfg.anchor_support_decay * self._to_float(anchor.support) + (1.0 - self.cfg.anchor_support_decay) * current
        return anchors

    def update_ttl(self, anchors: list[list[AnchorRecord]]) -> list[list[AnchorRecord]]:
        for batch_anchors in anchors:
            for anchor in batch_anchors:
                next_ttl = self._to_float(anchor.ttl) - 1.0
                anchor.ttl = max(next_ttl, 0.0)
        return anchors

    def apply_revision(
        self,
        anchors: list[list[AnchorRecord]],
        decisions: list[RevisionDecision],
    ) -> list[list[AnchorRecord]]:
        by_id = {decision.anchor_id: decision for decision in decisions}
        for batch_anchors in anchors:
            for anchor in batch_anchors:
                decision = by_id.get(anchor.id)
                if decision is None:
                    continue
                anchor.state = decision.new_state
                if decision.action == "retire":
                    anchor.viability = 0.0
                elif decision.action == "downgrade":
                    anchor.viability = min(self._to_float(anchor.viability), 0.5)
        return anchors

    def get_active_anchors(
        self,
        anchors: list[list[AnchorRecord]],
    ) -> list[list[AnchorRecord]]:
        active_states = {
            AnchorState.CANDIDATE,
            AnchorState.PROVISIONAL,
            AnchorState.CONFIRMED,
            AnchorState.DECAYING,
        }
        return [
            [anchor for anchor in batch_anchors if anchor.state in active_states]
            for batch_anchors in anchors
        ]

    def export_diagnostics(self, anchors: list[list[AnchorRecord]]) -> dict:
        flat = [anchor for batch in anchors for anchor in batch]
        if not flat:
            return {
                "num_active": 0,
                "state_counts": {state.value: 0 for state in AnchorState},
                "mean_anchor_score": 0.0,
                "mean_contradiction_pressure": 0.0,
                "mean_viability": 0.0,
                "mean_descendant_mass": 0.0,
                "mean_descendant_coherence": 0.0,
                "dead_end_count": 0,
            }

        state_counts = {state.value: 0 for state in AnchorState}
        for anchor in flat:
            state_counts[anchor.state.value] += 1

        return {
            "num_active": sum(anchor.state != AnchorState.DEAD_END for anchor in flat),
            "state_counts": state_counts,
            "mean_anchor_score": sum(self._to_float(anchor.score) for anchor in flat) / len(flat),
            "mean_contradiction_pressure": sum(self._to_float(anchor.contradiction_pressure) for anchor in flat) / len(flat),
            "mean_viability": sum(self._to_float(anchor.viability) for anchor in flat) / len(flat),
            "mean_descendant_mass": sum(self._to_float(anchor.descendant_mass or 0.0) for anchor in flat) / len(flat),
            "mean_descendant_coherence": sum(self._to_float(anchor.descendant_coherence or 0.0) for anchor in flat) / len(flat),
            "dead_end_count": state_counts[AnchorState.DEAD_END.value],
        }

    @staticmethod
    def _to_float(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)
