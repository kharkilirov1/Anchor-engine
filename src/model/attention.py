import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        B, T, _ = q.shape
        q = self.w_q(q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if causal:
            T_k = k.size(2)
            mask = torch.triu(
                torch.ones(T, T_k, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(out)


class AttentionResidual(nn.Module):
    """Replace standard residual with attention over previous layer outputs.
    Each layer learns input-dependent weights for aggregating previous representations.
    Solves PreNorm dilution (Kimi Team, 2026, arXiv:2603.15031).
    """

    def __init__(self, d_model: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, current: torch.Tensor, layer_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        n_prev = len(layer_outputs)
        if n_prev == 0:
            return self.layer_norm(current)

        # Stack previous outputs: [B, T, N, D]
        stacked = torch.stack(layer_outputs, dim=2)

        q = self.query_proj(current).unsqueeze(2)  # [B, T, 1, D]
        k = self.key_proj(stacked)  # [B, T, N, D]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(current.size(-1))
        weights = F.softmax(scores, dim=-1)  # [B, T, 1, N]

        aggregated = torch.matmul(weights, stacked).squeeze(2)  # [B, T, D]

        return self.layer_norm(current + aggregated)
