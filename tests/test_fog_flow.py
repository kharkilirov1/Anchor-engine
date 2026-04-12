from dataclasses import replace

import torch

from src.model.config import TOY_CONFIG
from src.model.fog_flow import (
    FogFlowBackbone,
    build_fog_geometries,
    resolve_fog_task_profile,
    select_fog_adapter_layers,
)


def test_build_fog_geometries_matches_adapter_layers() -> None:
    cfg = replace(TOY_CONFIG, n_layers=4, n_heads=2, fog_task_profile="stories")
    geoms = build_fog_geometries(cfg)
    assert len(geoms) == len(select_fog_adapter_layers(cfg, resolve_fog_task_profile(cfg)))
    assert len(geoms) == 2
    for geom in geoms:
        assert geom.d_compare % cfg.n_heads == 0
        assert geom.d_memory % cfg.n_heads == 0
        assert geom.d_expand >= cfg.d_model


def test_fog_flow_backbone_forward() -> None:
    cfg = replace(TOY_CONFIG, max_seq_len=24, use_fog_flow=True, fog_task_profile="code")
    model = FogFlowBackbone(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    out = model(x)
    assert out["hidden"].shape == (2, 24, cfg.d_model)
    assert len(out["layer_outputs"]) == cfg.n_layers + 1
    assert out["flow_type"] == "fog_hybrid"
    assert out["fog_profile"] == "code"
    assert out["fog_layers"] == [1, 2]


def test_fog_flow_respects_attention_residual_toggle() -> None:
    cfg_no_res = replace(TOY_CONFIG, use_fog_flow=True, use_attn_res=False)
    cfg_with_res = replace(TOY_CONFIG, use_fog_flow=True, use_attn_res=True)
    model_no_res = FogFlowBackbone(cfg_no_res)
    model_with_res = FogFlowBackbone(cfg_with_res)
    assert sum(p.numel() for p in model_with_res.parameters()) > sum(
        p.numel() for p in model_no_res.parameters()
    )
