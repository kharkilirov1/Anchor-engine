from __future__ import annotations

import torch

from src.fog.config import FOGConfig
from src.fog.model_fast import FastMotifTransformer, FastStructuredMotifTransformer


def _make_cfg() -> FOGConfig:
    return FOGConfig(
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


def test_fast_motif_forward_backward() -> None:
    cfg = _make_cfg()
    model = FastMotifTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()


def test_fast_structured_forward_backward() -> None:
    cfg = _make_cfg()
    model = FastStructuredMotifTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    assert len(out["geometry"]) == cfg.n_layers
    out["loss"].backward()


def test_fast_variants_use_grouped_kv() -> None:
    cfg = _make_cfg()
    motif = FastMotifTransformer(cfg)
    structured = FastStructuredMotifTransformer(cfg)
    assert motif.blocks[0].attn.kv_heads <= cfg.n_heads
    assert structured.blocks[0].attn.kv_heads <= cfg.n_heads
    assert motif.blocks[0].attn.kv_repeat * motif.blocks[0].attn.kv_heads == cfg.n_heads
