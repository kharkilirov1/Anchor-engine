from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fog.config import FOGConfig
from src.fog.model_runtime import RMSNorm, RuntimeStructuredAttention, RuntimeStructuredBlock, RuntimeStructuredFFN
from src.fog.model_structured_v2 import (
    LayerGeometryV2,
    StructuredMotifBlockV2,
    build_layer_geometries_v2,
)


class LocalSyntaxBranch(nn.Module):
    def __init__(self, d_model: int, d_local: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, d_local)
        self.depthwise = nn.Conv1d(
            in_channels=d_local,
            out_channels=d_local,
            kernel_size=kernel_size,
            groups=d_local,
            bias=True,
        )
        self.pointwise = nn.Conv1d(
            in_channels=d_local,
            out_channels=d_local,
            kernel_size=1,
            bias=True,
        )
        self.out_proj = nn.Linear(d_local, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        h = h.transpose(1, 2)
        h = F.pad(h, (self.kernel_size - 1, 0))
        h = self.depthwise(h)
        h = F.silu(self.pointwise(h))
        h = h.transpose(1, 2)
        h = self.drop(h)
        return self.out_proj(h)


class CodeAwareStructuredBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, geometry: LayerGeometryV2, dropout: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm_local = RMSNorm(d_model)
        self.attn = RuntimeStructuredAttention(d_model, geometry.d_compare, geometry.d_memory, n_heads)
        self.ffn = RuntimeStructuredFFN(d_model, geometry, dropout)
        d_local = max(32, min(d_model, geometry.d_compare * 2))
        kernel_size = 5 if geometry.stage != "late" else 3
        self.local_branch = LocalSyntaxBranch(
            d_model=d_model,
            d_local=d_local,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))
        self.local_scale = nn.Parameter(torch.tensor(0.07 if geometry.stage == "early" else 0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.drop(self.attn(self.norm1(x)))
        x = x + self.local_scale * self.drop(self.local_branch(self.norm_local(x)))
        x = x + self.ffn_scale * self.drop(self.ffn(self.norm2(x)))
        return x


class CodeAwareStructuredTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_geometries = build_layer_geometries_v2(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                CodeAwareStructuredBlock(
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


class LocalSyntaxBranchLight(nn.Module):
    def __init__(self, d_model: int, d_local: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, d_local)
        self.depthwise = nn.Conv1d(
            in_channels=d_local,
            out_channels=d_local,
            kernel_size=kernel_size,
            groups=d_local,
            bias=True,
        )
        self.out_proj = nn.Linear(d_local, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        h = h.transpose(1, 2)
        h = F.pad(h, (self.kernel_size - 1, 0))
        h = self.depthwise(h)
        h = h.transpose(1, 2)
        h = self.drop(F.silu(h))
        return self.out_proj(h)


class CodeAwareStructuredLightBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, geometry: LayerGeometryV2, dropout: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm_local = RMSNorm(d_model)
        self.attn = RuntimeStructuredAttention(d_model, geometry.d_compare, geometry.d_memory, n_heads)
        self.ffn = RuntimeStructuredFFN(d_model, geometry, dropout)
        self.use_local = geometry.stage == "early"
        if self.use_local:
            d_local = max(16, min(d_model // 4, geometry.d_compare))
            self.local_branch = LocalSyntaxBranchLight(
                d_model=d_model,
                d_local=d_local,
                kernel_size=3,
                dropout=dropout,
            )
            self.local_scale = nn.Parameter(torch.tensor(0.035))
        else:
            self.local_branch = None
            self.local_scale = None
        self.drop = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(geometry.residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.drop(self.attn(self.norm1(x)))
        if self.local_branch is not None and self.local_scale is not None:
            x = x + self.local_scale * self.drop(self.local_branch(self.norm_local(x)))
        x = x + self.ffn_scale * self.drop(self.ffn(self.norm2(x)))
        return x


class CodeAwareStructuredLightTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_geometries = build_layer_geometries_v2(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                CodeAwareStructuredLightBlock(
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


class RecentCopyBias(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, window: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.window = window
        self.copy_gate = nn.Linear(d_model, 1)
        self.lag_logits = nn.Parameter(torch.linspace(0.0, -1.5, steps=window))
        self.copy_scale = nn.Parameter(torch.tensor(1.25))

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        b, t, _ = hidden.shape
        gate = torch.sigmoid(self.copy_gate(hidden)).squeeze(-1)
        lag_weights = F.softmax(self.lag_logits, dim=0)
        bias = hidden.new_zeros((b, t, self.vocab_size))

        for lag in range(self.window):
            weight = lag_weights[lag]
            if lag == 0:
                tokens = input_ids
                contrib = gate * weight
                bias.scatter_add_(2, tokens.unsqueeze(-1), contrib.unsqueeze(-1))
            elif lag < t:
                tokens = input_ids[:, :-lag]
                contrib = gate[:, lag:] * weight
                bias[:, lag:, :].scatter_add_(2, tokens.unsqueeze(-1), contrib.unsqueeze(-1))

        return self.copy_scale * bias


class CodeAwareCopyTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig, copy_window: int = 8) -> None:
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
        self.copy_bias = RecentCopyBias(cfg.d_model, cfg.vocab_size, window=copy_window)

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
        logits = logits + self.copy_bias(x, input_ids)

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


class StructuredV2CopyTransformer(nn.Module):
    def __init__(self, cfg: FOGConfig, copy_window: int = 8) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_geometries = build_layer_geometries_v2(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                StructuredMotifBlockV2(
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
        self.copy_bias = RecentCopyBias(cfg.d_model, cfg.vocab_size, window=copy_window)
        self.copy_blend = nn.Parameter(torch.tensor(0.85))
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
        _, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        mask = self._causal_mask[:, :, :t, :t]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)
        logits = logits + self.copy_blend * self.copy_bias(x, input_ids)

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
