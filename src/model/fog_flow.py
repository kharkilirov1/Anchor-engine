from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbone import TransformerBlock
from src.model.config import ModelConfig


@dataclass(frozen=True)
class FogTaskProfile:
    name: str
    compare_ratio: float
    memory_ratio: float
    expand_ratio: float
    gate_ratio: float
    hybrid_start_ratio: float
    max_layers: int
    adapter_scale: float


@dataclass(frozen=True)
class FogLayerGeometry:
    layer_idx: int
    stage: str
    d_compare: int
    d_memory: int
    d_expand: int
    d_gate: int
    residual_scale: float


def _align_to_heads(value: int, n_heads: int) -> int:
    aligned = max(n_heads, (value // n_heads) * n_heads)
    if aligned < value:
        aligned += n_heads
    return aligned


def resolve_fog_task_profile(cfg: ModelConfig) -> FogTaskProfile:
    profile = cfg.fog_task_profile
    if profile == "stories":
        return FogTaskProfile(
            name="stories",
            compare_ratio=0.18,
            memory_ratio=0.60,
            expand_ratio=1.35,
            gate_ratio=0.08,
            hybrid_start_ratio=0.67,
            max_layers=2,
            adapter_scale=0.11,
        )
    if profile == "code":
        return FogTaskProfile(
            name="code",
            compare_ratio=0.34,
            memory_ratio=0.82,
            expand_ratio=1.70,
            gate_ratio=0.12,
            hybrid_start_ratio=0.45,
            max_layers=3,
            adapter_scale=0.16,
        )
    if profile == "math":
        return FogTaskProfile(
            name="math",
            compare_ratio=0.42,
            memory_ratio=0.70,
            expand_ratio=1.95,
            gate_ratio=0.14,
            hybrid_start_ratio=0.45,
            max_layers=3,
            adapter_scale=0.18,
        )
    if profile == "synthetic":
        return FogTaskProfile(
            name="synthetic",
            compare_ratio=0.32,
            memory_ratio=0.84,
            expand_ratio=2.10,
            gate_ratio=0.14,
            hybrid_start_ratio=0.0,
            max_layers=cfg.n_layers,
            adapter_scale=0.20,
        )
    return FogTaskProfile(
        name="balanced",
        compare_ratio=cfg.fog_compare_ratio,
        memory_ratio=cfg.fog_memory_ratio,
        expand_ratio=cfg.fog_expand_ratio,
        gate_ratio=cfg.fog_gate_ratio,
        hybrid_start_ratio=0.55,
        max_layers=min(2, cfg.n_layers),
        adapter_scale=0.13,
    )


def select_fog_adapter_layers(cfg: ModelConfig, profile: FogTaskProfile) -> list[int]:
    start_idx = min(cfg.n_layers - 1, max(0, int(cfg.n_layers * profile.hybrid_start_ratio)))
    candidate_layers = list(range(start_idx, cfg.n_layers))
    if len(candidate_layers) <= profile.max_layers:
        return candidate_layers
    return candidate_layers[-profile.max_layers:]


def build_fog_geometries(cfg: ModelConfig) -> list[FogLayerGeometry]:
    profile = resolve_fog_task_profile(cfg)
    adapter_layers = select_fog_adapter_layers(cfg, profile)
    geometries: list[FogLayerGeometry] = []
    if not adapter_layers:
        return geometries

    for adapter_pos, layer_idx in enumerate(adapter_layers):
        depth = adapter_pos / max(len(adapter_layers) - 1, 1)
        if depth < 0.34:
            stage = "early"
            compare_ratio = profile.compare_ratio * 0.95
            memory_ratio = profile.memory_ratio * 0.90
            expand_ratio = profile.expand_ratio * 0.90
            gate_ratio = profile.gate_ratio * 0.90
            residual_scale = profile.adapter_scale * 0.85
        elif depth < 0.67:
            stage = "middle"
            compare_ratio = profile.compare_ratio
            memory_ratio = profile.memory_ratio
            expand_ratio = profile.expand_ratio
            gate_ratio = profile.gate_ratio
            residual_scale = profile.adapter_scale
        else:
            stage = "late"
            compare_ratio = profile.compare_ratio * 1.05
            memory_ratio = profile.memory_ratio * 1.05
            expand_ratio = profile.expand_ratio * 1.10
            gate_ratio = profile.gate_ratio * 1.10
            residual_scale = profile.adapter_scale * 1.10

        geometries.append(
            FogLayerGeometry(
                layer_idx=layer_idx,
                stage=stage,
                d_compare=_align_to_heads(max(cfg.n_heads, int(cfg.d_model * compare_ratio)), cfg.n_heads),
                d_memory=_align_to_heads(max(cfg.n_heads, int(cfg.d_model * memory_ratio)), cfg.n_heads),
                d_expand=max(cfg.d_model, int(cfg.d_model * expand_ratio)),
                d_gate=max(4, int(cfg.d_model * gate_ratio)),
                residual_scale=residual_scale,
            )
        )
    return geometries


class FogAttention(nn.Module):
    def __init__(self, d_model: int, d_compare: int, d_memory: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_compare % n_heads == 0
        assert d_memory % n_heads == 0
        self.n_heads = n_heads
        self.compare_head_dim = d_compare // n_heads
        self.memory_head_dim = d_memory // n_heads
        self.d_memory = d_memory
        self.q_proj = nn.Linear(d_model, d_compare)
        self.k_proj = nn.Linear(d_model, d_compare)
        self.v_proj = nn.Linear(d_model, d_memory)
        self.out_proj = nn.Linear(d_memory, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.memory_head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.compare_head_dim)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_memory)
        return self.out_proj(y)


class FogFFN(nn.Module):
    def __init__(self, d_model: int, d_expand: int, d_gate: int, dropout: float) -> None:
        super().__init__()
        self.expand = nn.Linear(d_model, d_expand)
        self.gate = nn.Linear(d_model, d_gate)
        self.gate_up = nn.Linear(d_gate, d_expand)
        self.compress = nn.Linear(d_expand, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = F.silu(self.expand(x))
        gate = torch.sigmoid(self.gate_up(F.silu(self.gate(x))))
        return self.compress(self.dropout(expanded * gate))


class FogAdapterBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, geometry: FogLayerGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = FogAttention(cfg.d_model, geometry.d_compare, geometry.d_memory, cfg.n_heads, cfg.dropout)
        self.ffn = FogFFN(cfg.d_model, geometry.d_expand, geometry.d_gate, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_update = self.attn_scale * self.drop(self.attn(self.ln1(x), mask))
        ffn_update = self.ffn_scale * self.drop(self.ffn(self.ln2(x + attn_update)))
        return attn_update + ffn_update


class FogFlowBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.profile = resolve_fog_task_profile(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.layer_geometries = build_fog_geometries(cfg)
        self.fog_blocks = nn.ModuleDict(
            {str(geom.layer_idx): FogAdapterBlock(cfg, geom) for geom in self.layer_geometries}
        )
        self.fog_layers = [geom.layer_idx for geom in self.layer_geometries]
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor] | list[int] | str]:
        _, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        layer_outputs = [x]
        mask = self._causal_mask[:, :, :t, :t]
        for idx, block in enumerate(self.blocks):
            x = block(x, layer_outputs)
            if str(idx) in self.fog_blocks:
                x = x + self.fog_blocks[str(idx)](x, mask)
            layer_outputs.append(x)
        hidden = self.ln_final(x)
        return {
            "hidden": hidden,
            "layer_outputs": layer_outputs,
            "fog_layers": self.fog_layers,
            "fog_profile": self.profile.name,
            "flow_type": "fog_hybrid",
        }
