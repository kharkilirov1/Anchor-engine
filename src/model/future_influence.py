from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureInfluenceScorer(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        future_window: int = 16,
    ) -> dict[str, Any]:
        if not hidden.requires_grad:
            raise ValueError("hidden must require gradients for future influence scoring")

        if input_ids.size(1) < 2:
            zero_scores = torch.zeros(input_ids.shape, device=input_ids.device, dtype=hidden.dtype)
            return {
                "scores": zero_scores,
                "raw_scores": zero_scores,
                "loss": 0.0,
                "target_window": 0,
            }

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        if attention_mask is None:
            shift_mask = torch.ones_like(shift_labels, dtype=hidden.dtype)
        else:
            shift_mask = attention_mask[:, 1:].to(hidden.dtype)

        window = max(1, min(int(future_window), shift_labels.size(1)))
        target_logits = shift_logits[:, -window:, :]
        target_labels = shift_labels[:, -window:]
        target_mask = shift_mask[:, -window:]

        token_loss = F.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_labels.reshape(-1),
            reduction="none",
        ).view_as(target_labels)
        masked_loss = token_loss * target_mask
        loss = masked_loss.sum() / target_mask.sum().clamp_min(1.0)

        grad_hidden = torch.autograd.grad(loss, hidden, retain_graph=False, create_graph=False)[0]
        raw_scores = grad_hidden.norm(dim=-1)
        denom = raw_scores.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        norm_scores = raw_scores / denom
        if attention_mask is not None:
            norm_scores = norm_scores * attention_mask.to(norm_scores.dtype)
            raw_scores = raw_scores * attention_mask.to(raw_scores.dtype)

        return {
            "scores": norm_scores.detach(),
            "raw_scores": raw_scores.detach(),
            "loss": float(loss.detach().item()),
            "target_window": window,
        }
