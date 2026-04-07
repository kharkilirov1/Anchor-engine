"""Motif-Aware Transformer — heterogeneous internal subspaces per FOG hypothesis."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fog.config import FOGConfig


class MotifAwareAttention(nn.Module):
    """Q/K in narrow compare-space, V in wide memory-space."""

    def __init__(self, d_model: int, d_compare: int, d_memory: int, n_heads: int):
        super().__init__()
        assert d_compare % n_heads == 0
        assert d_memory % n_heads == 0
        self.n_heads = n_heads
        self.compare_head_dim = d_compare // n_heads
        self.memory_head_dim = d_memory // n_heads
        self.d_compare = d_compare
        self.d_memory = d_memory

        # Φ(proj): compare subspace
        self.q_proj = nn.Linear(d_model, d_compare)
        self.k_proj = nn.Linear(d_model, d_compare)
        # Φ(memory): memory subspace
        self.v_proj = nn.Linear(d_model, d_memory)
        # back to residual
        self.out_proj = nn.Linear(d_memory, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.compare_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.memory_head_dim).transpose(1, 2)

        # Φ(compare): scoring in narrow space
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.compare_head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Φ(select): softmax gating
        attn = F.softmax(scores, dim=-1)

        # Φ(aggregate): memory-space carries content
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_memory)
        return self.out_proj(y)


class MotifAwareFFN(nn.Module):
    """Expand in wide space, gate in narrow control space, compress back."""

    def __init__(self, d_model: int, d_expand: int, d_gate: int):
        super().__init__()
        # Φ(expand): into rich intermediate
        self.expand = nn.Linear(d_model, d_expand)
        # Φ(select/control): narrow gating path
        self.gate = nn.Linear(d_model, d_gate)
        self.gate_up = nn.Linear(d_gate, d_expand)
        # Φ(compress): back to residual
        self.compress = nn.Linear(d_expand, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        gate = torch.sigmoid(self.gate_up(F.silu(self.gate(x))))
        return self.compress(F.silu(expanded * gate))


class MotifAwareBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_compare: int,
        d_memory: int,
        d_expand: int,
        d_gate: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MotifAwareAttention(d_model, d_compare, d_memory, n_heads)
        self.ffn = MotifAwareFFN(d_model, d_expand, d_gate)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class MotifTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            MotifAwareBlock(
                d_model=cfg.d_model,
                d_compare=cfg.d_compare,
                d_memory=cfg.d_memory,
                d_expand=cfg.d_expand,
                d_gate=cfg.d_gate,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

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

        return {"logits": logits, "loss": loss}
