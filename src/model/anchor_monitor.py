from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.anchor_types import AnchorRecord
from src.model.config import ModelConfig


class ContradictionMonitor(nn.Module):
    _REGIME_COMPATIBILITY: dict[int, set[int]] = {
        11: {11, 13, 16},
        21: {14, 15, 21, 22, 23},
        31: {15, 31, 32, 33},
        41: {41, 42, 43, 44},
        51: {15, 51, 52, 53},
    }

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        hidden: torch.Tensor,
        anchors: list[list[AnchorRecord]],
        aux: dict | None = None,
    ) -> dict:
        aux = aux or {}
        input_ids: torch.Tensor | None = aux.get("input_ids")

        pressure_by_anchor: dict[int, float] = {}
        pressure_components: dict[int, dict[str, float]] = {}

        for batch_idx, batch_anchors in enumerate(anchors):
            seq_hidden = hidden[batch_idx]
            seq_ids = None if input_ids is None else input_ids[batch_idx]
            T = seq_hidden.size(0)
            for anchor in batch_anchors:
                span_len = max(anchor.end_idx - anchor.start_idx + 1, 1)
                horizon = max(int(float(anchor.ttl) * 4), span_len * 4)
                start = min(anchor.end_idx + 1, T)
                stop = min(start + horizon, T)
                if start >= stop:
                    hidden_contradiction = 0.0
                    token_contradiction = 0.0
                    pattern_contradiction = 0.0
                    future_shift = 0.0
                    similarity = 1.0
                    descendant_mass = 0.0
                    descendant_coherence = 0.0
                else:
                    future = seq_hidden[start:stop]
                    mean_future = future.mean(dim=0, keepdim=True)
                    similarity = float(F.cosine_similarity(anchor.repr.unsqueeze(0), mean_future, dim=-1).mean().item())
                    future_shift = float((future - anchor.repr.unsqueeze(0)).norm(dim=-1).mean().item())
                    hidden_contradiction = max(0.0, (1.0 - similarity) / 2.0)

                    if seq_ids is None:
                        token_contradiction = hidden_contradiction
                        pattern_contradiction = hidden_contradiction
                        descendant_mass = max(0.0, 1.0 - hidden_contradiction)
                        descendant_coherence = max(0.0, similarity)
                    else:
                        anchor_token = int(seq_ids[anchor.end_idx].item())
                        future_tokens = seq_ids[start:stop]
                        match_ratio = float((future_tokens == anchor_token).float().mean().item())
                        token_contradiction = 1.0 - match_ratio
                        pattern_contradiction, descendant_mass, descendant_coherence = self._pattern_stats(
                            seq_ids,
                            anchor,
                            start,
                            stop,
                        )

                if seq_ids is None:
                    contradiction = hidden_contradiction
                else:
                    contradiction = (
                        0.20 * hidden_contradiction
                        + 0.20 * token_contradiction
                        + 0.60 * pattern_contradiction
                    )
                contradiction = float(max(0.0, min(1.0, contradiction)))

                anchor.contradiction_pressure = contradiction
                anchor.descendant_mass = descendant_mass
                anchor.descendant_coherence = descendant_coherence
                pressure_by_anchor[anchor.id] = contradiction
                pressure_components[anchor.id] = {
                    "future_shift": future_shift,
                    "local_similarity": similarity,
                    "hidden_contradiction": hidden_contradiction,
                    "token_contradiction": token_contradiction,
                    "pattern_contradiction": pattern_contradiction,
                    "descendant_mass": descendant_mass,
                    "descendant_coherence": descendant_coherence,
                    "self_contradiction": contradiction,
                }

        return {
            "contradiction_pressure": pressure_by_anchor,
            "pressure_components": pressure_components,
        }

    @staticmethod
    def _pattern_stats(
        seq_ids: torch.Tensor,
        anchor: AnchorRecord,
        start: int,
        stop: int,
    ) -> tuple[float, float, float]:
        anchor_span = seq_ids[anchor.start_idx: anchor.end_idx + 1]
        span_len = anchor_span.numel()
        future_tokens = seq_ids[start:stop]
        if future_tokens.numel() < span_len:
            return 1.0, 0.0, 0.0

        sims: list[float] = []
        root_token = int(anchor_span[0].item())
        root_hits = []
        regime_hits = []
        pos_weights = torch.linspace(1.0, 0.4, steps=span_len, device=anchor_span.device)
        pos_weights = pos_weights / pos_weights.sum()
        allowed_tokens = ContradictionMonitor._REGIME_COMPATIBILITY.get(root_token)
        for offset in range(0, future_tokens.numel() - span_len + 1):
            window = future_tokens[offset: offset + span_len]
            exact_match = float((window == anchor_span).float().mean().item())
            overlap = len(set(window.tolist()) & set(anchor_span.tolist())) / max(len(set(anchor_span.tolist())), 1)
            root_persistence = float((window == root_token).float().mean().item())
            positional_match = float(((window == anchor_span).float() * pos_weights).sum().item())
            if allowed_tokens is None:
                regime_compatibility = overlap
            else:
                regime_compatibility = sum(int(int(token.item()) in allowed_tokens) for token in window) / max(span_len, 1)
            similarity = (
                0.25 * exact_match
                + 0.15 * overlap
                + 0.35 * positional_match
                + 0.25 * regime_compatibility
            )
            sims.append(similarity)
            root_hits.append(root_persistence)
            regime_hits.append(regime_compatibility)

        best_similarity = max(sims) if sims else 0.0
        mean_root_persistence = sum(root_hits) / max(len(root_hits), 1)
        mean_regime_compatibility = sum(regime_hits) / max(len(regime_hits), 1)
        descendant_mass = sum(sim >= 0.6 for sim in sims) / max(len(sims), 1)
        descendant_coherence = (
            0.55 * (sum(sims) / max(len(sims), 1))
            + 0.15 * mean_root_persistence
            + 0.30 * mean_regime_compatibility
        )
        pattern_contradiction = 1.0 - (
            0.60 * best_similarity
            + 0.10 * mean_root_persistence
            + 0.30 * mean_regime_compatibility
        )
        return pattern_contradiction, float(descendant_mass), float(descendant_coherence)
