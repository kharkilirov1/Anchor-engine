from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fog.config import FOGConfig
from src.fog.model_structured_v2 import LayerGeometryV2, build_layer_geometries_v2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class RuntimeStructuredAttention(nn.Module):
    def __init__(self, d_model: int, d_compare: int, d_memory: int, n_heads: int) -> None:
        super().__init__()
        assert d_compare % n_heads == 0
        assert d_memory % n_heads == 0
        self.n_heads = n_heads
        self.compare_head_dim = d_compare // n_heads
        self.memory_head_dim = d_memory // n_heads
        self.d_compare = d_compare
        self.d_memory = d_memory
        self.in_proj = nn.Linear(d_model, (2 * d_compare) + d_memory)
        self.out_proj = nn.Linear(d_memory, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        packed = self.in_proj(x)
        q, k, v = packed.split([self.d_compare, self.d_compare, self.d_memory], dim=-1)
        q = q.view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.memory_head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_memory)
        return self.out_proj(y)


class RuntimeStructuredFFN(nn.Module):
    def __init__(self, d_model: int, geometry: LayerGeometryV2, dropout: float) -> None:
        super().__init__()
        self.stage = geometry.stage
        self.d_expand = geometry.d_expand
        self.d_gate = geometry.d_gate
        self.fused_in = nn.Linear(d_model, geometry.d_expand + geometry.d_gate)
        self.gate_up = nn.Linear(geometry.d_gate, geometry.d_expand)
        self.compress = nn.Linear(geometry.d_expand, d_model)
        self.drop = nn.Dropout(dropout)

        if self.stage == "middle":
            self.stage_proj = nn.Linear(geometry.d_expand, geometry.d_expand)
            self.stage_scale = 0.35
        elif self.stage == "late":
            self.stage_proj = nn.Linear(geometry.d_expand, geometry.d_expand)
            self.stage_scale = 0.25
        else:
            self.stage_proj = None
            self.stage_scale = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded, gate_seed = self.fused_in(x).split([self.d_expand, self.d_gate], dim=-1)
        h = F.silu(expanded)
        gate_hidden = F.silu(gate_seed)
        h = h * torch.sigmoid(self.gate_up(gate_hidden))
        if self.stage_proj is not None:
            if self.stage == "middle":
                h = h + self.stage_scale * F.silu(self.stage_proj(h))
            else:
                h = h + self.stage_scale * torch.tanh(self.stage_proj(h))
        h = self.drop(h)
        return self.compress(h)


class RuntimeStructuredBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, geometry: LayerGeometryV2, dropout: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = RuntimeStructuredAttention(d_model, geometry.d_compare, geometry.d_memory, n_heads)
        self.ffn = RuntimeStructuredFFN(d_model, geometry, dropout)
        self.drop = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.drop(self.attn(self.norm1(x)))
        x = x + self.ffn_scale * self.drop(self.ffn(self.norm2(x)))
        return x


class RuntimeStructuredMotifTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_geometries = build_layer_geometries_v2(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                RuntimeStructuredBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    geometry=geometry,
                    dropout=cfg.dropout,
                )
                for geometry in self.layer_geometries
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[dict[str, int | str]]]:
        _, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
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
