from __future__ import annotations

import torch

from src.model.anchor_types import AnchorRecord, AnchorState
from src.model.qwen_generation_bias import (
    compute_anchor_generation_gate,
    compute_anchor_logits_bias,
)


def test_compute_anchor_generation_gate_increases_with_drift() -> None:
    small = compute_anchor_generation_gate(
        similarity=0.50,
        support=0.8,
        contradiction_pressure=0.8,
        viability=0.8,
        conflict_threshold=0.55,
    )
    large = compute_anchor_generation_gate(
        similarity=0.10,
        support=0.8,
        contradiction_pressure=0.8,
        viability=0.8,
        conflict_threshold=0.55,
    )
    assert large > small > 0.0


def test_compute_anchor_logits_bias_returns_zero_without_active_gate() -> None:
    projection = torch.nn.Linear(4, 6, bias=False)
    last_hidden = torch.ones(1, 4)
    anchor = AnchorRecord(
        id=1,
        start_idx=0,
        end_idx=1,
        repr=torch.ones(4),
        score=0.9,
        state=AnchorState.CONFIRMED,
        support=0.9,
        contradiction_pressure=0.8,
        viability=0.8,
        ttl=4.0,
    )
    bias, diagnostics = compute_anchor_logits_bias(
        last_hidden=last_hidden,
        active_anchors=[anchor],
        output_projection=projection,
        conflict_threshold=0.0,
        bias_scale=1.0,
    )
    assert torch.allclose(bias, torch.zeros_like(bias))
    assert diagnostics == []


def test_compute_anchor_logits_bias_emits_signal_when_similarity_low() -> None:
    projection = torch.nn.Linear(4, 6, bias=False)
    last_hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    anchor = AnchorRecord(
        id=1,
        start_idx=0,
        end_idx=1,
        repr=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        score=0.9,
        state=AnchorState.CONFIRMED,
        support=0.9,
        contradiction_pressure=0.9,
        viability=0.9,
        ttl=4.0,
    )
    bias, diagnostics = compute_anchor_logits_bias(
        last_hidden=last_hidden,
        active_anchors=[anchor],
        output_projection=projection,
        conflict_threshold=0.55,
        bias_scale=1.0,
    )
    assert bias.shape == (1, 6)
    assert diagnostics
    assert diagnostics[0]["gate"] > 0.0
