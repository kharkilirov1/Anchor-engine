#!/usr/bin/env python3
"""Local smoke test for bias parameter changes.

Run this after modifying bias parameters to verify basic functionality.
Uses small model (Qwen2.5-0.5B) for faster testing.
"""

from __future__ import annotations

import sys
import torch
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.model.qwen_generation_bias import (
    get_bias_domain_profile,
    build_bias_token_weights,
    compute_entropy_conflict_bias_scale,
)


def test_domain_profiles() -> None:
    """Verify domain profiles have expected values."""
    print("Testing domain profiles...")
    
    # Math domain
    math_profile = get_bias_domain_profile(
        "Prove sqrt(2) is irrational by contradiction"
    )
    assert math_profile.name == "math"
    assert 0.30 <= math_profile.alpha_multiplier <= 0.40, \
        f"Math alpha {math_profile.alpha_multiplier} outside conservative range"
    assert math_profile.hard_block_forbidden is True, \
        "Math should have hard_block enabled"
    print(f"  Math: alpha={math_profile.alpha_multiplier}, hard_block={math_profile.hard_block_forbidden}")
    
    # Code domain
    code_profile = get_bias_domain_profile(
        "FastAPI async service with Pydantic models"
    )
    assert code_profile.name == "code"
    assert code_profile.alpha_multiplier >= 0.80
    assert code_profile.hard_block_forbidden is True
    print(f"  Code: alpha={code_profile.alpha_multiplier}, hard_block={code_profile.hard_block_forbidden}")
    
    # Vegan domain
    vegan_profile = get_bias_domain_profile(
        "Vegan chef meal plan"
    )
    assert vegan_profile.name == "vegan"
    assert 0.30 <= vegan_profile.alpha_multiplier <= 0.35
    assert vegan_profile.hard_block_forbidden is True
    print(f"  Vegan: alpha={vegan_profile.alpha_multiplier}, hard_block={vegan_profile.hard_block_forbidden}")
    
    print("  [OK] All profiles correct\n")


def test_bias_weights() -> None:
    """Test bias token weight construction."""
    print("Testing bias token weights...")
    
    class FakeTokenizer:
        def __init__(self):
            self.vocab = {
                0: " the",
                1: " a",
                2: "egg",
                3: "vegan",
                4: "tofu",
                5: "django",
            }
        
        def __call__(self, text, **kwargs):
            text = str(text).lower()
            ids = [k for k, v in self.vocab.items() if v.strip() in text]
            if not ids:
                ids = [0]
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        
        def convert_ids_to_tokens(self, ids):
            return [self.vocab.get(i, f"tok{i}") for i in ids]
    
    tokenizer = FakeTokenizer()
    device = torch.device("cpu")
    
    # Vegan prompt
    weights, blocked_ids, diag = build_bias_token_weights(
        tokenizer=tokenizer,
        vocab_size=6,
        device=device,
        prompt="You are a vegan chef",
    )
    
    assert diag["domain"] == "vegan"
    assert 2 in blocked_ids, "'egg' should be blocked"
    assert 5 not in blocked_ids, "'django' not relevant for vegan"
    assert weights[2] == 0.0, "Blocked terms have weight 0"
    assert weights[3] >= 1.0, "'vegan' should be boosted"
    print(f"  Vegan: blocked={blocked_ids}, boosted={[i for i, w in enumerate(weights) if w > 1]}")
    print("  [OK] Weight construction correct\n")


def test_entropy_pressure_gating() -> None:
    """Test that gating behaves correctly."""
    print("Testing entropy-pressure gating...")
    
    # High pressure, low entropy (should still have moderate alpha due to rescue_floor)
    result = compute_entropy_conflict_bias_scale(
        normalized_entropy=0.1,  # Low entropy (confident model)
        contradiction_pressure=0.8,  # High pressure (needs rescue)
        alpha_max=1.0,
        entropy_threshold=0.40,
        pressure_threshold=0.60,
        entropy_slope=0.08,
        pressure_slope=0.08,
        pressure_rescue_floor=0.20,
    )
    
    # With rescue_floor=0.20, even low entropy should give alpha >= 0.20 * pressure_gate
    assert result["alpha_t"] > 0.15, \
        f"High pressure should give meaningful alpha even with low entropy, got {result['alpha_t']}"
    assert result["pressure_gate"] > 0.5, "High pressure should open pressure_gate"
    print(f"  High pressure + low entropy: alpha={result['alpha_t']:.3f}")
    
    # Low pressure, high entropy (should have low alpha)
    result2 = compute_entropy_conflict_bias_scale(
        normalized_entropy=0.9,
        contradiction_pressure=0.2,
        alpha_max=1.0,
        entropy_threshold=0.40,
        pressure_threshold=0.60,
        entropy_slope=0.08,
        pressure_slope=0.08,
        pressure_rescue_floor=0.20,
    )
    assert result2["alpha_t"] < 0.3, "Low pressure should give low alpha"
    print(f"  Low pressure + high entropy: alpha={result2['alpha_t']:.3f}")
    print("  [OK] Gating behavior correct\n")


def test_revision_path_enabled() -> None:
    """Verify that revision path is enabled in overlay."""
    print("Testing revision path...")
    
    from src.model.config import TOY_CONFIG
    
    class DummyModel:
        class config:
            hidden_size = 32
            vocab_size = 100
            max_position_embeddings = 64
        
        def get_output_embeddings(self):
            return torch.nn.Linear(32, 100)
    
    overlay = QwenAnchorOverlay(
        base_model=DummyModel(),
        cfg=TOY_CONFIG,
    )
    
    # Check that _build_base_arbiter method exists and is callable
    assert hasattr(overlay, '_build_base_arbiter'), "Missing _build_base_arbiter method"
    
    # Test arbiter construction with dummy anchors
    from src.model.anchor_types import AnchorRecord, AnchorState
    
    dummy_anchors = [[
        AnchorRecord(
            id=1,
            start_idx=0,
            end_idx=5,
            repr=torch.randn(32),
            score=0.9,
            state=AnchorState.CONFIRMED,
            support=0.9,
            contradiction_pressure=0.8,  # High pressure
            viability=0.3,  # Low viability
            ttl=5.0,
        ),
        AnchorRecord(
            id=2,
            start_idx=10,
            end_idx=15,
            repr=torch.randn(32),
            score=0.9,
            state=AnchorState.CONFIRMED,
            support=0.9,
            contradiction_pressure=0.3,  # Low pressure
            viability=0.8,  # High viability
            ttl=5.0,
        ),
    ]]
    
    arbiter = overlay._build_base_arbiter(dummy_anchors, revision_threshold=0.5)
    
    # Only anchor 1 should trigger revision (high pressure + low viability)
    assert 1 in arbiter, "High-pressure anchor should be in arbiter"
    assert 2 not in arbiter, "Low-pressure anchor should not be in arbiter"
    assert arbiter[1]["proposal_type"] == "pressure_triggered_revision"
    print(f"  Arbiter keys: {list(arbiter.keys())}")
    print("  [OK] Revision path enabled\n")


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("ABPT Bias Changes - Local Smoke Test")
    print("=" * 60 + "\n")
    
    try:
        test_domain_profiles()
        test_bias_weights()
        test_entropy_pressure_gating()
        test_revision_path_enabled()
        
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
