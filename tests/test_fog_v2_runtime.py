from __future__ import annotations

import torch

from src.fog.config import FOGConfig
from src.fog.model_runtime import RuntimeStructuredMotifTransformer
from src.fog.model_structured_v2 import StructuredMotifTransformerV2, build_layer_geometries_v2


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


def test_build_layer_geometries_v2_has_depth_stages() -> None:
    cfg = _make_cfg()
    geometries = build_layer_geometries_v2(cfg)
    assert len(geometries) == cfg.n_layers
    assert geometries[0].stage == "early"
    assert any(g.stage == "middle" for g in geometries)
    assert geometries[-1].stage == "late"
    assert geometries[0].residual_scale < geometries[-1].residual_scale


def test_structured_v2_forward_backward() -> None:
    cfg = _make_cfg()
    model = StructuredMotifTransformerV2(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    assert len(out["geometry"]) == cfg.n_layers
    out["loss"].backward()


def test_runtime_structured_forward_backward() -> None:
    cfg = _make_cfg()
    model = RuntimeStructuredMotifTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(x, y)
    assert out["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    assert len(out["geometry"]) == cfg.n_layers
    out["loss"].backward()
