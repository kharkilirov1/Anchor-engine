from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.anchor_types import AnchorRecord
from src.model.config import ModelConfig


class ContradictionMonitor(nn.Module):
    _REGIME_ROOT_ALIAS: dict[int, int] = {
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

    @staticmethod
    def _cosine01(a: torch.Tensor, b: torch.Tensor) -> float:
        cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).mean()
        return float(((cosine + 1.0) * 0.5).item())

    def _domain_mode(self) -> str:
        if self.cfg.anchor_domain_mode in {"synthetic", "real"}:
            return self.cfg.anchor_domain_mode
        return "real"

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

        domain_mode = self._domain_mode()

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
                    elif domain_mode == "synthetic":
                        anchor_token = int(seq_ids[anchor.end_idx].item())
                        future_tokens = seq_ids[start:stop]
                        match_ratio = float((future_tokens == anchor_token).float().mean().item())
                        token_contradiction = 1.0 - match_ratio
                        pattern_contradiction, descendant_mass, descendant_coherence = self._pattern_stats_synthetic(
                            seq_ids,
                            anchor,
                            start,
                            stop,
                        )
                    else:
                        token_contradiction, pattern_contradiction, descendant_mass, descendant_coherence = (
                            self._pattern_stats_real(
                                seq_hidden=seq_hidden,
                                seq_ids=seq_ids,
                                anchor=anchor,
                                start=start,
                                stop=stop,
                            )
                        )

                if seq_ids is None:
                    contradiction = hidden_contradiction
                elif domain_mode == "synthetic":
                    contradiction = (
                        0.20 * hidden_contradiction
                        + 0.20 * token_contradiction
                        + 0.60 * pattern_contradiction
                    )
                else:
                    contradiction = (
                        0.55 * hidden_contradiction
                        + 0.15 * token_contradiction
                        + 0.30 * pattern_contradiction
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
    def _pattern_stats_synthetic(
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
        root_token = ContradictionMonitor.infer_reference_root(anchor_span)
        root_hits = []
        regime_hits = []
        pos_weights = torch.linspace(1.0, 0.4, steps=span_len, device=anchor_span.device)
        pos_weights = pos_weights / pos_weights.sum()
        for offset in range(0, future_tokens.numel() - span_len + 1):
            window = future_tokens[offset: offset + span_len]
            exact_match = float((window == anchor_span).float().mean().item())
            overlap = len(set(window.tolist()) & set(anchor_span.tolist())) / max(len(set(anchor_span.tolist())), 1)
            if root_token is None:
                root_persistence = overlap
            else:
                root_persistence = float((window == root_token).float().mean().item())
            positional_match = float(((window == anchor_span).float() * pos_weights).sum().item())
            regime_compatibility = ContradictionMonitor.regime_compatibility_score(
                window_tokens=window,
                anchor_span=anchor_span,
                root_token=root_token,
            )
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
            0.60 * (sum(sims) / max(len(sims), 1))
            + 0.25 * mean_root_persistence
            + 0.15 * mean_regime_compatibility
        )
        pattern_contradiction = 1.0 - (
            0.60 * best_similarity
            + 0.20 * mean_root_persistence
            + 0.20 * mean_regime_compatibility
        )
        return pattern_contradiction, float(descendant_mass), float(descendant_coherence)

    def _pattern_stats_real(
        self,
        seq_hidden: torch.Tensor,
        seq_ids: torch.Tensor,
        anchor: AnchorRecord,
        start: int,
        stop: int,
    ) -> tuple[float, float, float, float]:
        anchor_hidden_span = seq_hidden[anchor.start_idx: anchor.end_idx + 1]
        anchor_token_span = seq_ids[anchor.start_idx: anchor.end_idx + 1]
        span_len = anchor_hidden_span.size(0)
        future_hidden = seq_hidden[start:stop]
        future_tokens = seq_ids[start:stop]
        if future_hidden.size(0) < span_len:
            return 1.0, 1.0, 0.0, 0.0

        anchor_mean = anchor_hidden_span.mean(dim=0)
        anchor_delta = anchor_hidden_span[1:] - anchor_hidden_span[:-1]
        sims: list[float] = []
        token_overlaps: list[float] = []
        transition_sims: list[float] = []
        for offset in range(0, future_hidden.size(0) - span_len + 1):
            window_hidden = future_hidden[offset: offset + span_len]
            window_tokens = future_tokens[offset: offset + span_len]
            mean_sim = self._cosine01(anchor_mean, window_hidden.mean(dim=0))
            if anchor_delta.numel() > 0:
                window_delta = window_hidden[1:] - window_hidden[:-1]
                transition_sim = self._cosine01(anchor_delta.flatten(), window_delta.flatten())
            else:
                transition_sim = mean_sim
            token_overlap = len(set(window_tokens.tolist()) & set(anchor_token_span.tolist())) / max(
                len(set(anchor_token_span.tolist())),
                1,
            )
            similarity = 0.55 * mean_sim + 0.25 * transition_sim + 0.20 * token_overlap
            sims.append(similarity)
            token_overlaps.append(token_overlap)
            transition_sims.append(transition_sim)

        best_similarity = max(sims) if sims else 0.0
        mean_similarity = sum(sims) / max(len(sims), 1)
        mean_overlap = sum(token_overlaps) / max(len(token_overlaps), 1)
        mean_transition = sum(transition_sims) / max(len(transition_sims), 1)
        descendant_mass = sum(sim >= 0.68 for sim in sims) / max(len(sims), 1)
        descendant_coherence = 0.65 * mean_similarity + 0.20 * mean_transition + 0.15 * mean_overlap
        pattern_contradiction = 1.0 - (0.75 * best_similarity + 0.25 * mean_similarity)
        token_contradiction = 1.0 - max(mean_overlap, best_similarity)
        return (
            float(token_contradiction),
            float(pattern_contradiction),
            float(descendant_mass),
            float(descendant_coherence),
        )

    @classmethod
    def resolve_regime_root_from_ids(
        cls,
        seq_ids: torch.Tensor,
        span_start: int,
        span_end: int,
    ) -> int | None:
        span_tokens = seq_ids[span_start: span_end + 1]
        return cls.resolve_regime_root_from_span(span_tokens)

    @classmethod
    def resolve_regime_root_from_span(
        cls,
        span_tokens: torch.Tensor,
    ) -> int | None:
        for token in span_tokens.tolist():
            root = cls._REGIME_ROOT_ALIAS.get(int(token))
            if root is not None:
                return root
        return None

    @classmethod
    def infer_reference_root(
        cls,
        span_tokens: torch.Tensor,
    ) -> int | None:
        root = cls.resolve_regime_root_from_span(span_tokens)
        if root is not None:
            return root
        if span_tokens.numel() == 0:
            return None
        unique_tokens, counts = torch.unique(span_tokens, return_counts=True)
        return int(unique_tokens[torch.argmax(counts)].item())

    @classmethod
    def regime_compatibility_score(
        cls,
        window_tokens: torch.Tensor,
        anchor_span: torch.Tensor,
        root_token: int | None,
    ) -> float:
        if window_tokens.numel() == 0:
            return 0.0

        window_list = [int(token) for token in window_tokens.tolist()]
        anchor_token_set = {int(token) for token in anchor_span.tolist()}
        overlap = len(set(window_list) & anchor_token_set) / max(len(anchor_token_set), 1)
        exact_match = float((window_tokens == anchor_span).float().mean().item())

        if root_token is None:
            root_persistence = overlap
            alias_compatibility = 0.0
        else:
            root_persistence = sum(token == root_token for token in window_list) / max(len(window_list), 1)
            alias_compatibility = sum(
                cls._REGIME_ROOT_ALIAS.get(token) == root_token for token in window_list
            ) / max(len(window_list), 1)

        allowed_tokens = cls._REGIME_COMPATIBILITY.get(root_token) if root_token is not None else None
        if allowed_tokens is None:
            return float(
                0.45 * exact_match
                + 0.30 * overlap
                + 0.10 * alias_compatibility
                + 0.15 * root_persistence
            )

        hard_compatibility = sum(token in allowed_tokens for token in window_list) / max(len(window_list), 1)
        return float(
            0.55 * hard_compatibility
            + 0.20 * overlap
            + 0.15 * alias_compatibility
            + 0.10 * root_persistence
        )
