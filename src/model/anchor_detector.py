from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model.anchor_types import AnchorCandidate
from src.model.config import ModelConfig


class AnchorDetector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.prior_head = nn.Linear(cfg.d_model, 1)
        self.prior_weight = cfg.anchor_prior_weight
        self.runtime_weight = cfg.anchor_runtime_weight
        self.threshold = cfg.anchor_threshold
        self.max_candidates = cfg.anchor_max_candidates

    @staticmethod
    def _standardize(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (values - mean) / std

    def forward(
        self,
        hidden: torch.Tensor,
        history: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        del attention_mask

        if history is None:
            shifted = torch.roll(hidden, shifts=1, dims=1)
            shifted[:, 0] = hidden[:, 0]
        else:
            shifted = history

        delta = hidden - shifted
        runtime_raw = delta.norm(dim=-1) / math.sqrt(hidden.size(-1))
        runtime_logits = self._standardize(runtime_raw)
        runtime_score = torch.sigmoid(runtime_logits)

        prior_logits = self.prior_head(hidden).squeeze(-1)
        prior_logits = self._standardize(prior_logits)
        prior_score = torch.sigmoid(prior_logits)

        combined_logits = self.prior_weight * prior_logits + self.runtime_weight * runtime_logits
        scores = torch.sigmoid(combined_logits)
        semantic_weights = combined_logits

        B, T, _ = hidden.shape
        positions = torch.arange(T, device=hidden.device)
        starts = torch.clamp(positions - 1, min=0)
        span_bounds = torch.stack((starts, positions), dim=-1).unsqueeze(0).expand(B, -1, -1)

        candidates: list[list[AnchorCandidate]] = []
        for b in range(B):
            batch_scores = scores[b]
            peak_mask = torch.zeros(T, dtype=torch.bool, device=hidden.device)
            for t in range(T):
                left = batch_scores[t - 1] if t > 0 else batch_scores[t]
                right = batch_scores[t + 1] if t + 1 < T else batch_scores[t]
                peak_mask[t] = batch_scores[t] >= left and batch_scores[t] >= right

            valid_idx = torch.nonzero((batch_scores >= self.threshold) & peak_mask, as_tuple=False).flatten()
            if valid_idx.numel() > self.max_candidates:
                top_idx = torch.topk(batch_scores[valid_idx], k=self.max_candidates).indices
                valid_idx = valid_idx[top_idx]
                valid_idx, _ = torch.sort(valid_idx)

            batch_candidates: list[AnchorCandidate] = []
            for t_tensor in valid_idx:
                t = int(t_tensor.item())
                batch_candidates.append(
                    AnchorCandidate(
                        start_idx=int(span_bounds[b, t, 0].item()),
                        end_idx=int(span_bounds[b, t, 1].item()),
                        repr=hidden[b, t],
                        score=scores[b, t],
                        semantic_weight=semantic_weights[b, t],
                    )
                )
            candidates.append(batch_candidates)

        return {
            "candidates": candidates,
            "scores": scores,
            "span_bounds": span_bounds,
            "semantic_weights": semantic_weights,
            "prior_scores": prior_score,
            "runtime_scores": runtime_score,
        }
