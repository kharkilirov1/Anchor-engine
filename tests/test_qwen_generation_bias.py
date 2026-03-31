from __future__ import annotations

import pytest
import torch

from src.model.anchor_types import AnchorRecord, AnchorState
from src.model.qwen_generation_bias import (
    apply_frequency_penalty,
    apply_forbidden_token_penalty,
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    build_bias_token_weights,
    compute_anchor_generation_gate,
    compute_entropy_conflict_bias_scale,
    compute_anchor_logits_bias,
    compute_normalized_entropy,
    get_bias_domain_profile,
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


def test_compute_anchor_logits_bias_can_mask_blocked_tokens() -> None:
    projection = torch.nn.Linear(4, 6, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(6, 4)[:6])
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
    weights = torch.ones(6)
    weights[1] = 0.0
    bias, diagnostics = compute_anchor_logits_bias(
        last_hidden=last_hidden,
        active_anchors=[anchor],
        output_projection=projection,
        conflict_threshold=0.55,
        bias_scale=1.0,
        bias_token_weights=weights,
    )
    assert diagnostics
    assert torch.isclose(bias[0, 1], torch.tensor(0.0))


def test_build_bias_token_weights_returns_domain_profile() -> None:
    class _ToyTokenizer:
        def __init__(self) -> None:
            self.surfaces = {
                0: " the",
                1: " theory",
                2: " vegan",
                3: " milk",
                4: " tofu",
                5: " django",
            }

        def __call__(self, texts, padding=True, truncation=True, max_length=16, return_tensors="pt", add_special_tokens=False):
            del padding, truncation, return_tensors, add_special_tokens
            rows = []
            for text in texts if isinstance(texts, list) else [texts]:
                lowered = str(text).lower()
                ids = [token_id for token_id, surface in self.surfaces.items() if surface.strip() in lowered]
                if not ids:
                    ids = [0]
                rows.append(torch.tensor(ids, dtype=torch.long))
            padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
            mask = (padded != 0).long()
            return {"input_ids": padded, "attention_mask": mask}

        def convert_ids_to_tokens(self, ids):
            return [self.surfaces.get(int(token_id), f"tok{token_id}") for token_id in ids]

    tokenizer = _ToyTokenizer()
    weights, blocked_ids, diag = build_bias_token_weights(
        tokenizer=tokenizer,
        vocab_size=6,
        device=torch.device("cpu"),
        prompt="You are a vegan chef. Write a weekly meal plan.",
    )
    assert weights is not None
    assert weights.shape == (6,)
    assert diag["domain"] == "vegan"
    assert weights[0] < 0.1
    assert weights[3] == 0.0  # Blocked terms have weight 0.0 (hard block via apply_forbidden_token_penalty)
    assert weights[4] > 1.0
    assert 3 in blocked_ids
    assert diag["masked_token_fraction"] >= 0.0


def test_get_bias_domain_profile_detects_math_prompt() -> None:
    profile = get_bias_domain_profile("Prove that sqrt(2)+sqrt(3) is irrational by contradiction.")
    assert profile.name == "math"
    assert 0.55 <= profile.alpha_multiplier <= 0.65  # Tuned 0.60 for valid math proofs


def test_apply_forbidden_token_penalty_suppresses_blocked_ids() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0]], dtype=torch.float32)
    adjusted = apply_forbidden_token_penalty(
        logits=logits,
        forbidden_token_ids={1, 3},
        penalty=5.0,
    )
    assert torch.isclose(adjusted[0, 0], logits[0, 0])
    assert adjusted[0, 1] == pytest.approx(-4.0)
    assert adjusted[0, 3] == pytest.approx(-5.0)


def test_compute_normalized_entropy_stays_between_zero_and_one() -> None:
    logits = torch.tensor([[10.0, -10.0, -10.0], [0.0, 0.0, 0.0]], dtype=torch.float32)

    entropy = compute_normalized_entropy(logits)

    assert torch.all(entropy >= 0.0)
    assert torch.all(entropy <= 1.0 + 1e-6)
    assert entropy[1] > entropy[0]


def test_compute_normalized_entropy_can_use_top_k_slice() -> None:
    logits = torch.tensor([[5.0, 4.0, 1.0, -3.0]], dtype=torch.float32)

    entropy_top2 = compute_normalized_entropy(logits, top_k=2)
    entropy_full = compute_normalized_entropy(logits)

    assert entropy_top2.shape == entropy_full.shape
    assert float(entropy_top2.item()) <= 1.0 + 1e-6


def test_compute_normalized_entropy_sanitizes_non_finite_logits() -> None:
    logits = torch.tensor([[float("nan"), 0.0, float("inf"), float("-inf")]], dtype=torch.float32)

    entropy = compute_normalized_entropy(logits, top_k=3)

    assert torch.isfinite(entropy).all()
    assert 0.0 <= float(entropy.item()) <= 1.0 + 1e-6


def test_compute_entropy_conflict_bias_scale_requires_uncertainty_and_pressure() -> None:
    low = compute_entropy_conflict_bias_scale(
        normalized_entropy=0.10,
        contradiction_pressure=0.10,
        alpha_max=1.5,
        entropy_threshold=0.40,
        pressure_threshold=0.30,
        entropy_slope=0.10,
        pressure_slope=0.10,
        pressure_rescue_floor=0.20,
    )
    high = compute_entropy_conflict_bias_scale(
        normalized_entropy=0.85,
        contradiction_pressure=0.80,
        alpha_max=1.5,
        entropy_threshold=0.40,
        pressure_threshold=0.30,
        entropy_slope=0.10,
        pressure_slope=0.10,
        pressure_rescue_floor=0.20,
    )

    assert high["alpha_t"] > low["alpha_t"]
    assert high["entropy_gate"] > low["entropy_gate"]
    assert high["pressure_gate"] > low["pressure_gate"]


def test_compute_entropy_conflict_bias_scale_keeps_pressure_floor_when_entropy_low() -> None:
    low_entropy_high_pressure = compute_entropy_conflict_bias_scale(
        normalized_entropy=0.05,
        contradiction_pressure=0.95,
        alpha_max=1.5,
        entropy_threshold=0.35,
        pressure_threshold=0.60,
        entropy_slope=0.08,
        pressure_slope=0.08,
        pressure_rescue_floor=0.20,
    )

    assert low_entropy_high_pressure["pressure_gate"] > 0.9
    assert low_entropy_high_pressure["alpha_t"] > 0.0


def test_compute_entropy_conflict_bias_scale_sanitizes_non_finite_inputs() -> None:
    diag = compute_entropy_conflict_bias_scale(
        normalized_entropy=float("nan"),
        contradiction_pressure=float("nan"),
        alpha_max=1.5,
        entropy_threshold=0.35,
        pressure_threshold=0.60,
        entropy_slope=0.08,
        pressure_slope=0.08,
        pressure_rescue_floor=0.20,
    )

    assert diag["entropy_input_isfinite"] == 0.0
    assert diag["pressure_input_isfinite"] == 0.0
    assert diag["alpha_isfinite"] == 1.0
    assert diag["alpha_t"] >= 0.0
    assert torch.isfinite(torch.tensor(diag["alpha_t"]))


def test_apply_repetition_penalty_downweights_seen_positive_logits() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0]])
    generated_ids = torch.tensor([[0, 2, 0]])
    adjusted = apply_repetition_penalty(logits=logits, generated_ids=generated_ids, penalty=1.2)
    assert adjusted[0, 0] < logits[0, 0]
    assert adjusted[0, 2] < logits[0, 2]
    assert adjusted[0, 1] == logits[0, 1]


def test_apply_frequency_penalty_tracks_repeat_count() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.5]])
    generated_ids = torch.tensor([[1, 1, 2]])
    adjusted = apply_frequency_penalty(logits=logits, generated_ids=generated_ids, penalty=0.25)
    assert torch.isclose(adjusted[0, 1], torch.tensor(0.5))
    assert torch.isclose(adjusted[0, 2], torch.tensor(0.25))
    assert torch.isclose(adjusted[0, 0], logits[0, 0])


def test_apply_no_repeat_ngram_blocks_repeated_trigram_tail() -> None:
    logits = torch.zeros(1, 6)
    generated_ids = torch.tensor([[1, 2, 3, 1, 2]])
    adjusted, blocked = apply_no_repeat_ngram(
        logits=logits,
        generated_ids=generated_ids,
        ngram_size=3,
    )
    assert 3 in blocked
    assert adjusted[0, 3] < -1e20
