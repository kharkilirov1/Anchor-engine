import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.attention import MultiHeadAttention, AttentionResidual


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_attn_res = cfg.use_attn_res

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

        if self.use_attn_res:
            self.attn_res = AttentionResidual(cfg.d_model, layer_idx)
        else:
            self.ln_res = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, layer_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out = self.attn(normed, normed, normed, causal=True)

        if self.use_attn_res:
            x = self.attn_res(attn_out, layer_outputs)
        else:
            x = self.ln_res(x + attn_out)

        x = x + self.ff(self.ln2(x))
        return x


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.ln_final = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor) -> dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        layer_outputs = [x]
        for block in self.blocks:
            x = block(x, layer_outputs)
            layer_outputs.append(x)

        hidden = self.ln_final(x)
        return {"hidden": hidden, "layer_outputs": layer_outputs}
