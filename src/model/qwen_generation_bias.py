from __future__ import annotations

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
