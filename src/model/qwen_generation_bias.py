from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch
import torch.nn.functional as F

from src.model.anchor_types import AnchorRecord


def compute_anchor_generation_gate(
    similarity: float,
    support: float,
    contradiction_pressure: float,
    viability: float,
    conflict_threshold: float,
) -> float:
    drift = max(0.0, float(conflict_threshold) - float(similarity))
    return (
        drift
        * max(0.0, float(support))
        * (0.20 + 0.80 * float(contradiction_pressure))
        * (0.55 + 0.45 * float(viability))
    )


def compute_anchor_logits_bias(
    last_hidden: torch.Tensor,
    active_anchors: Iterable[AnchorRecord],
    output_projection: torch.nn.Module,
    conflict_threshold: float,
    bias_scale: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if last_hidden.ndim != 2 or last_hidden.size(0) != 1:
        raise ValueError("last_hidden must be shaped [1, hidden_dim]")

    dtype = last_hidden.dtype
    device = last_hidden.device
    bias: torch.Tensor | None = None
    diagnostics: list[dict[str, float]] = []
    current = F.normalize(last_hidden, dim=-1)

    for anchor in active_anchors:
        anchor_repr = anchor.repr.to(device=device, dtype=dtype).unsqueeze(0)
        similarity = float(F.cosine_similarity(current, F.normalize(anchor_repr, dim=-1), dim=-1).item())
        gate = compute_anchor_generation_gate(
            similarity=similarity,
            support=float(anchor.support),
            contradiction_pressure=float(anchor.contradiction_pressure),
            viability=float(anchor.viability),
            conflict_threshold=conflict_threshold,
        )
        if gate <= 0.0:
            continue
        anchor_logits = output_projection(anchor_repr).squeeze(0)
        anchor_logits = anchor_logits - anchor_logits.mean()
        anchor_logits = anchor_logits / anchor_logits.std().clamp_min(1e-6)
        scaled = float(bias_scale) * float(gate) * anchor_logits
        bias = scaled if bias is None else bias + scaled
        diagnostics.append(
            {
                "anchor_id": float(anchor.id),
                "similarity": float(similarity),
                "gate": float(gate),
                "support": float(anchor.support),
                "contradiction_pressure": float(anchor.contradiction_pressure),
                "viability": float(anchor.viability),
            }
        )

    if bias is None:
        output_dim = int(output_projection.weight.shape[0])
        bias = torch.zeros(output_dim, device=device, dtype=dtype)
    return bias.unsqueeze(0), diagnostics


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("logits must be shaped [1, vocab_size]")
    if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
        raise ValueError("generated_ids must be shaped [1, seq_len]")
    if penalty <= 1.0:
        return logits

    adjusted = logits.clone()
    token_ids = generated_ids[0].tolist()
    for token_id in set(int(token_id) for token_id in token_ids):
        value = adjusted[0, token_id]
        adjusted[0, token_id] = torch.where(
            value > 0,
            value / penalty,
            value * penalty,
        )
    return adjusted


def apply_frequency_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("logits must be shaped [1, vocab_size]")
    if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
        raise ValueError("generated_ids must be shaped [1, seq_len]")
    if penalty <= 0.0:
        return logits

    adjusted = logits.clone()
    counts = Counter(int(token_id) for token_id in generated_ids[0].tolist())
    for token_id, count in counts.items():
        adjusted[0, token_id] = adjusted[0, token_id] - (float(count) * penalty)
    return adjusted


def _collect_blocked_tokens_for_ngram(
    token_ids: list[int],
    ngram_size: int,
) -> set[int]:
    if ngram_size <= 1 or len(token_ids) < ngram_size - 1:
        return set()

    prefix = tuple(token_ids[-(ngram_size - 1) :])
    blocked: set[int] = set()
    for idx in range(len(token_ids) - ngram_size + 1):
        ngram = token_ids[idx : idx + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            blocked.add(int(ngram[-1]))
    return blocked


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    ngram_size: int,
) -> tuple[torch.Tensor, set[int]]:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("logits must be shaped [1, vocab_size]")
    if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
        raise ValueError("generated_ids must be shaped [1, seq_len]")
    if ngram_size <= 1:
        return logits, set()

    blocked = _collect_blocked_tokens_for_ngram(
        token_ids=[int(token_id) for token_id in generated_ids[0].tolist()],
        ngram_size=ngram_size,
    )
    if not blocked:
        return logits, blocked

    adjusted = logits.clone()
    for token_id in blocked:
        adjusted[0, token_id] = torch.finfo(adjusted.dtype).min
    return adjusted, blocked
