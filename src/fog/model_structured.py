"""Structured motif-aware transformer closer to the FOG hypothesis.

This variant goes beyond a single repeated motif-aware block and introduces:
1. depth-wise geometry changes (early / middle / late motif stages),
2. explicit stage-specialized feed-forward computations,
3. learnable residual scales to stabilize heterogeneous training.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fog.config import FOGConfig


@dataclass(frozen=True)
class LayerGeometry:
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


def build_layer_geometries(cfg: FOGConfig) -> list[LayerGeometry]:
    """Construct a simple early/middle/late morphology schedule.

    Early layers bias toward projection/compression.
    Middle layers bias toward memory/compose.
    Late layers bias toward expand/transform.
    """
    geoms: list[LayerGeometry] = []
    n_layers = cfg.n_layers
    for idx in range(n_layers):
        depth_pos = idx / max(n_layers - 1, 1)
        if depth_pos < 0.34:
            stage = "early"
            d_compare = _align_to_heads(max(cfg.d_compare, cfg.d_model // 4), cfg.n_heads)
            d_memory = _align_to_heads(max(cfg.n_heads * 12, int(cfg.d_memory * 0.75)), cfg.n_heads)
            d_expand = max(cfg.d_gate * 4, int(cfg.d_expand * 0.75))
            d_gate = max(cfg.d_gate, cfg.d_model // 10)
            residual_scale = 0.12
        elif depth_pos < 0.67:
            stage = "middle"
            d_compare = _align_to_heads(max(cfg.d_compare, cfg.d_model // 4), cfg.n_heads)
            d_memory = _align_to_heads(max(cfg.d_memory, int(cfg.d_model * 0.875)), cfg.n_heads)
            d_expand = max(cfg.d_expand, int(cfg.d_model * 2.25))
            d_gate = max(cfg.d_gate, cfg.d_model // 8)
            residual_scale = 0.16
        else:
            stage = "late"
            d_compare = _align_to_heads(max(cfg.d_compare, cfg.d_model // 3), cfg.n_heads)
            d_memory = _align_to_heads(max(cfg.n_heads * 12, int(cfg.d_memory * 0.875)), cfg.n_heads)
            d_expand = max(cfg.d_expand, int(cfg.d_expand * 1.25))
            d_gate = max(cfg.d_gate, cfg.d_model // 6)
            residual_scale = 0.18

        geoms.append(
            LayerGeometry(
                stage=stage,
                d_compare=d_compare,
                d_memory=d_memory,
                d_expand=d_expand,
                d_gate=d_gate,
                residual_scale=residual_scale,
            )
        )
    return geoms


class StructuredAttention(nn.Module):
    def __init__(self, d_model: int, d_compare: int, d_memory: int, n_heads: int) -> None:
        super().__init__()
        assert d_compare % n_heads == 0
        assert d_memory % n_heads == 0
        self.n_heads = n_heads
        self.compare_head_dim = d_compare // n_heads
        self.memory_head_dim = d_memory // n_heads
        self.d_compare = d_compare
        self.d_memory = d_memory

        self.q_proj = nn.Linear(d_model, d_compare)
        self.k_proj = nn.Linear(d_model, d_compare)
        self.v_proj = nn.Linear(d_model, d_memory)
        self.out_proj = nn.Linear(d_memory, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.memory_head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.compare_head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_memory)
        return self.out_proj(y)


class StructuredMotifFFN(nn.Module):
    def __init__(self, d_model: int, geometry: LayerGeometry, dropout: float) -> None:
        super().__init__()
        self.stage = geometry.stage
        self.expand = nn.Linear(d_model, geometry.d_expand)
        self.gate = nn.Linear(d_model, geometry.d_gate)
        self.gate_up = nn.Linear(geometry.d_gate, geometry.d_expand)
        self.drop = nn.Dropout(dropout)

        if self.stage == "middle":
            self.compose_proj = nn.Linear(geometry.d_expand, geometry.d_expand)
            self.transform_proj = None
        elif self.stage == "late":
            self.compose_proj = None
            self.transform_proj = nn.Linear(geometry.d_expand, geometry.d_expand)
        else:
            self.compose_proj = None
            self.transform_proj = None

        self.compress = nn.Linear(geometry.d_expand, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = F.silu(self.expand(x))
        gate = torch.sigmoid(self.gate_up(F.silu(self.gate(x))))
        h = expanded * gate

        if self.compose_proj is not None:
            h = h + 0.5 * F.silu(self.compose_proj(h))
        if self.transform_proj is not None:
            h = F.silu(self.transform_proj(h))

        h = self.drop(h)
        return self.compress(h)


class StructuredMotifBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, geometry: LayerGeometry, dropout: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = StructuredAttention(
            d_model=d_model,
            d_compare=geometry.d_compare,
            d_memory=geometry.d_memory,
            n_heads=n_heads,
        )
        self.ffn = StructuredMotifFFN(d_model=d_model, geometry=geometry, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn_scale * self.drop(self.attn(self.ln1(x), mask))
        x = x + self.ffn_scale * self.drop(self.ffn(self.ln2(x)))
        return x


class StructuredMotifTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_geometries = build_layer_geometries(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                StructuredMotifBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    geometry=geometry,
                    dropout=cfg.dropout,
                )
                for geometry in self.layer_geometries
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[dict[str, int | str]]]:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        mask = self._causal_mask[:, :, :t, :t]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            if loss_mask is not None:
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                flat_mask = loss_mask.view(-1).bool()
                if flat_mask.any():
                    loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
                else:
                    loss = torch.tensor(0.0, device=logits.device)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        geometry_summary = [
            {
                "stage": g.stage,
                "d_compare": g.d_compare,
                "d_memory": g.d_memory,
                "d_expand": g.d_expand,
                "d_gate": g.d_gate,
            }
            for g in self.layer_geometries
        ]
        return {"logits": logits, "loss": loss, "geometry": geometry_summary}
