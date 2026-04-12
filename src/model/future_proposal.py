from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig


@dataclass
class FutureProposalCandidate:
    start: int
    end: int
    repr: torch.Tensor
    score: torch.Tensor
    root_token: int | None


class FutureProposalHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hidden_dim = max(32, int(cfg.anchor_future_proposal_hidden))
        self.score_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.repr_delta = nn.Sequential(
            nn.Linear(cfg.d_model * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.d_model),
        )

    @staticmethod
    def _cosine01_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).mean()
        return (cosine + 1.0) * 0.5

    def _candidate_lengths(
        self,
        span_len: int,
        available: int,
    ) -> list[int]:
        if available <= 0:
            return []
        lengths = {
            max(1, span_len // 2),
            max(1, span_len),
            max(1, min(available, span_len + max(1, span_len // 2))),
            max(1, min(available, span_len * 2)),
        }
        return sorted(length for length in lengths if 1 <= length <= available)

    def _search_bounds(
        self,
        anchor,
        seq_len: int,
    ) -> tuple[int, int, int]:
        span_len = max(int(anchor.end_idx) - int(anchor.start_idx) + 1, 1)
        start = min(int(anchor.end_idx) + 1, seq_len)
        if start >= seq_len:
            return start, start, span_len
        base_horizon = max(
            int(float(anchor.ttl) * float(self.cfg.anchor_future_proposal_horizon_scale)),
            int(span_len * float(self.cfg.anchor_future_proposal_span_scale)),
        )
        horizon = min(max(base_horizon, span_len), int(self.cfg.anchor_future_proposal_max_horizon))
        stop = min(seq_len, start + max(horizon, 1))
        return start, stop, span_len

    def _subsample_candidates(
        self,
        candidates: list[FutureProposalCandidate],
    ) -> list[FutureProposalCandidate]:
        max_windows = max(1, int(self.cfg.anchor_future_proposal_max_windows))
        if len(candidates) <= max_windows:
            return candidates
        idx = torch.linspace(0, len(candidates) - 1, steps=max_windows).round().long().tolist()
        return [candidates[i] for i in idx]

    def _build_candidates(
        self,
        seq_hidden: torch.Tensor,
        seq_ids: torch.Tensor | None,
        anchor,
    ) -> list[FutureProposalCandidate]:
        seq_len = seq_hidden.size(0)
        start, stop, span_len = self._search_bounds(anchor, seq_len)
        if stop <= start:
            return []

        available = stop - start
        lengths = self._candidate_lengths(span_len, available)
        if not lengths:
            return []

        anchor_hidden_span = seq_hidden[int(anchor.start_idx): int(anchor.end_idx) + 1]
        anchor_delta = (
            anchor_hidden_span[1:] - anchor_hidden_span[:-1]
            if anchor_hidden_span.size(0) > 1
            else None
        )

        candidates: list[FutureProposalCandidate] = []
        for length in lengths:
            max_offset = stop - length + 1
            for offset in range(start, max_offset):
                window_hidden = seq_hidden[offset: offset + length]
                window_mean = window_hidden.mean(dim=0)
                mean_sim = self._cosine01_tensor(anchor.repr, window_mean)
                contrast = 1.0 - mean_sim

                if anchor_delta is not None and anchor_delta.numel() > 0 and window_hidden.size(0) > 1:
                    window_delta = window_hidden[1:] - window_hidden[:-1]
                    transition_sim = self._cosine01_tensor(anchor_delta.mean(dim=0), window_delta.mean(dim=0))
                else:
                    transition_sim = mean_sim

                coherence = ((F.cosine_similarity(window_hidden, window_mean.unsqueeze(0), dim=-1) + 1.0) * 0.5).mean()

                tail_hidden = seq_hidden[offset + length: stop]
                if tail_hidden.numel() > 0:
                    tail_support = self._cosine01_tensor(window_mean, tail_hidden.mean(dim=0))
                else:
                    tail_support = coherence

                if seq_ids is None:
                    token_overlap = seq_hidden.new_tensor(0.0)
                    root_token = None
                else:
                    anchor_ids = seq_ids[int(anchor.start_idx): int(anchor.end_idx) + 1]
                    window_ids = seq_ids[offset: offset + length]
                    anchor_token_set = {int(token) for token in anchor_ids.tolist()}
                    window_token_set = {int(token) for token in window_ids.tolist()}
                    token_overlap = seq_hidden.new_tensor(
                        len(anchor_token_set & window_token_set) / max(len(anchor_token_set), 1)
                    )
                    root_token = int(window_ids[-1].item())

                distance = max(0, offset - int(anchor.end_idx))
                distance_decay = seq_hidden.new_tensor(1.0 / (1.0 + distance / max(float(span_len), 1.0)))
                pressure = seq_hidden.new_tensor(float(anchor.contradiction_pressure))
                viability_gap = seq_hidden.new_tensor(1.0 - float(anchor.viability))
                descendant_gap = seq_hidden.new_tensor(1.0 - float(anchor.descendant_coherence or 0.0))

                conflict_signal = 0.55 * contrast + 0.25 * (1.0 - transition_sim) + 0.20 * (1.0 - token_overlap)
                plausibility = 0.45 * coherence + 0.35 * tail_support + 0.20 * distance_decay
                repair_readiness = 0.60 * pressure + 0.40 * viability_gap
                if float(conflict_signal.item()) < 0.18 or float(repair_readiness.item()) < 0.35:
                    continue

                feature_vec = torch.stack(
                    [
                        contrast,
                        mean_sim,
                        transition_sim,
                        coherence,
                        tail_support,
                        token_overlap,
                        distance_decay,
                        pressure,
                        viability_gap,
                        descendant_gap,
                    ],
                    dim=0,
                ).to(device=seq_hidden.device, dtype=seq_hidden.dtype)
                learned_logit = 0.25 * self.score_mlp(feature_vec.unsqueeze(0)).squeeze(0).squeeze(-1)
                heuristic_logit = (
                    2.4 * (conflict_signal - 0.35)
                    + 2.0 * (plausibility - 0.55)
                    + 1.4 * (repair_readiness - 0.50)
                    + 0.5 * (descendant_gap - 0.35)
                )
                score = torch.sigmoid(
                    (heuristic_logit + learned_logit) / max(float(self.cfg.anchor_future_proposal_temperature), 1e-6)
                )
                candidates.append(
                    FutureProposalCandidate(
                        start=offset,
                        end=offset + length - 1,
                        repr=window_mean,
                        score=score,
                        root_token=root_token,
                    )
                )

        return self._subsample_candidates(candidates)

    def propose(
        self,
        seq_hidden: torch.Tensor,
        seq_ids: torch.Tensor | None,
        anchor,
    ) -> dict | None:
        candidates = self._build_candidates(seq_hidden=seq_hidden, seq_ids=seq_ids, anchor=anchor)
        if not candidates:
            return None

        scores = torch.stack([candidate.score for candidate in candidates], dim=0)
        best_score, best_idx = scores.max(dim=0)
        if float(best_score.item()) < float(self.cfg.anchor_future_proposal_threshold):
            return None

        topk = min(int(self.cfg.anchor_future_proposal_topk), len(candidates))
        top_scores, top_idx = torch.topk(scores, k=topk)
        top_weights = torch.softmax(
            top_scores / max(float(self.cfg.anchor_future_proposal_temperature), 1e-6),
            dim=0,
        )
        top_repr = torch.stack([candidates[int(idx.item())].repr for idx in top_idx], dim=0)
        anchor_repr = anchor.repr.unsqueeze(0).expand_as(top_repr)
        fusion_in = torch.cat(
            [anchor_repr, top_repr, top_repr - anchor_repr, top_repr * anchor_repr],
            dim=-1,
        )
        fused_repr = top_repr + float(self.cfg.anchor_future_proposal_residual_scale) * self.repr_delta(fusion_in)
        proposal_repr = (top_weights.unsqueeze(-1) * fused_repr).sum(dim=0)

        best_candidate = candidates[int(best_idx.item())]
        return {
            "repr": proposal_repr,
            "proposal_type": "future_window_head",
            "proposal_score": float(best_score.item()),
            "proposal_score_tensor": best_score,
            "proposal_span": (best_candidate.start, best_candidate.end),
            "proposal_root_token": best_candidate.root_token,
            "proposal_candidate_count": len(candidates),
        }
