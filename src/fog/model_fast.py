"""Faster FOG variants that keep the same test protocol.

Design goals:
1. keep motif-aware geometry,
2. reduce CPU cost through fused projections,
3. use grouped KV heads,
4. replace expensive stage-specific expand-space transforms with cheap low-rank adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fog.config import FOGConfig
from src.fog.model_structured import LayerGeometry, build_layer_geometries


def _choose_kv_heads(n_heads: int) -> int:
    if n_heads % 4 == 0:
        return max(1, n_heads // 4)
    if n_heads % 2 == 0:
        return max(1, n_heads // 2)
    return 1


class FastGroupedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_compare: int,
        d_memory: int,
        n_heads: int,
        kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        assert d_compare % n_heads == 0
        assert d_memory % n_heads == 0
        self.n_heads = n_heads
        self.kv_heads = kv_heads or _choose_kv_heads(n_heads)
        assert n_heads % self.kv_heads == 0
        self.kv_repeat = n_heads // self.kv_heads
        self.compare_head_dim = d_compare // n_heads
        self.memory_head_dim = d_memory // n_heads
        self.d_compare = d_compare
        self.d_memory = d_memory

        total_out = (
            d_compare
            + self.kv_heads * self.compare_head_dim
            + self.kv_heads * self.memory_head_dim
        )
        self.in_proj = nn.Linear(d_model, total_out)
        self.out_proj = nn.Linear(d_memory, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        packed = self.in_proj(x)

        q_end = self.d_compare
        k_end = q_end + self.kv_heads * self.compare_head_dim
        q, k, v = packed.split(
            [self.d_compare, self.kv_heads * self.compare_head_dim, self.kv_heads * self.memory_head_dim],
            dim=-1,
        )

        q = q.view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.compare_head_dim).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.memory_head_dim).transpose(1, 2)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.compare_head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_memory)
        return self.out_proj(y)


class FastMotifFFN(nn.Module):
    def __init__(self, d_model: int, d_expand: int, d_gate: int, dropout: float) -> None:
        super().__init__()
        self.fused_in = nn.Linear(d_model, d_expand + d_gate)
        self.gate_up = nn.Linear(d_gate, d_expand)
        self.compress = nn.Linear(d_expand, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.fused_in(x)
        expanded, gate_seed = packed.split([self.compress.in_features, self.gate_up.in_features], dim=-1)
        expanded = F.silu(expanded)
        gate = torch.sigmoid(self.gate_up(F.silu(gate_seed)))
        h = self.drop(expanded * gate)
        return self.compress(h)


class FastMotifBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_compare: int,
        d_memory: int,
        d_expand: int,
        d_gate: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = FastGroupedAttention(
            d_model=d_model,
            d_compare=d_compare,
            d_memory=d_memory,
            n_heads=n_heads,
            kv_heads=n_heads,
        )
        self.ffn = FastMotifFFN(d_model, d_expand, d_gate, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class FastMotifTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                FastMotifBlock(
                    d_model=cfg.d_model,
                    d_compare=cfg.d_compare,
                    d_memory=cfg.d_memory,
                    d_expand=cfg.d_expand,
                    d_gate=cfg.d_gate,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.drop = nn.Dropout(cfg.dropout)
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
    ) -> dict[str, torch.Tensor]:
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
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask]) if flat_mask.any() else torch.tensor(0.0, device=logits.device)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return {"logits": logits, "loss": loss}


class FastStructuredFFN(nn.Module):
    def __init__(self, d_model: int, geometry: LayerGeometry, dropout: float) -> None:
        super().__init__()
        self.stage = geometry.stage
        self.d_expand = geometry.d_expand
        self.d_gate = geometry.d_gate
        self.fused_in = nn.Linear(d_model, geometry.d_expand + geometry.d_gate)
        self.gate_up = nn.Linear(geometry.d_gate, geometry.d_expand)
        self.compress = nn.Linear(geometry.d_expand, d_model)
        self.drop = nn.Dropout(dropout)

        if self.stage in ("middle", "late"):
            self.stage_adapter = nn.Linear(geometry.d_gate, geometry.d_expand)
            self.stage_scale = nn.Parameter(torch.tensor(0.10 if self.stage == "middle" else 0.08))
        else:
            self.stage_adapter = None
            self.stage_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.fused_in(x)
        expanded, gate_seed = packed.split([self.d_expand, self.d_gate], dim=-1)
        expanded = F.silu(expanded)
        gate_hidden = F.silu(gate_seed)
        gate = torch.sigmoid(self.gate_up(gate_hidden))
        h = expanded * gate

        if self.stage_adapter is not None and self.stage_scale is not None:
            h = h + self.stage_scale * torch.tanh(self.stage_adapter(gate_hidden))

        h = self.drop(h)
        return self.compress(h)


class FastStructuredBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, geometry: LayerGeometry, dropout: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = FastGroupedAttention(
            d_model=d_model,
            d_compare=geometry.d_compare,
            d_memory=geometry.d_memory,
            n_heads=n_heads,
            kv_heads=_choose_kv_heads(n_heads),
        )
        self.ffn = FastStructuredFFN(d_model=d_model, geometry=geometry, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn_scale * self.drop(self.attn(self.ln1(x), mask))
        x = x + self.ffn_scale * self.drop(self.ffn(self.ln2(x)))
        return x


class FastStructuredMotifTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_geometries = build_layer_geometries(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                FastStructuredBlock(
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
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask]) if flat_mask.any() else torch.tensor(0.0, device=logits.device)
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
