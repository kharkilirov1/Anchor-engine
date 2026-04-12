from __future__ import annotations

import torch

from src.fog.config import FOGConfig
from src.fog.model_code import (
    CodeAwareCopyTransformer,
    CodeAwareStructuredLightTransformer,
    CodeAwareStructuredTransformer,
    StructuredV2CopyTransformer,
)


def test_code_aware_structured_forward_backward() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=6,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = CodeAwareStructuredTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    assert len(out["geometry"]) == cfg.n_layers
    out["loss"].backward()


def test_code_aware_structured_has_local_branch() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = CodeAwareStructuredTransformer(cfg)
    block = model.blocks[0]
    assert block.local_branch.kernel_size == 5
    assert block.local_scale.item() > 0.0


def test_code_aware_structured_light_forward_backward() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = CodeAwareStructuredLightTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()


def test_code_aware_structured_light_uses_local_only_early() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=6,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = CodeAwareStructuredLightTransformer(cfg)
    assert model.blocks[0].local_branch is not None
    assert model.blocks[-1].local_branch is None


def test_code_aware_copy_forward_backward() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = CodeAwareCopyTransformer(cfg, copy_window=4)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()


def test_code_aware_copy_has_recent_bias_module() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = CodeAwareCopyTransformer(cfg, copy_window=6)
    assert model.copy_bias.window == 6
    assert model.copy_bias.vocab_size == cfg.vocab_size


def test_structured_v2_copy_forward_backward() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = StructuredV2CopyTransformer(cfg, copy_window=4)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    assert len(out["geometry"]) == cfg.n_layers
    out["loss"].backward()


def test_structured_v2_copy_has_copy_controls() -> None:
    cfg = FOGConfig(
        vocab_size=64,
        d_model=96,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
        d_ff=256,
        d_compare=24,
        d_memory=72,
        d_expand=192,
        d_gate=16,
    )
    model = StructuredV2CopyTransformer(cfg, copy_window=6)
    assert model.copy_bias.window == 6
    assert model.copy_blend.item() > 0.0
