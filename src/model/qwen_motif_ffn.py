from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.model.qwen_motif_router import BaseMotifRouter


class QwenMotifSplitMLP(nn.Module):
    def __init__(
        self,
        base_mlp: nn.Module,
        motif_index: torch.Tensor,
        router: BaseMotifRouter,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.gate_proj = self._require_module(base_mlp, "gate_proj")
        self.up_proj = self._require_module(base_mlp, "up_proj")
        self.down_proj = self._require_module(base_mlp, "down_proj")
        self.act_fn = self._require_attr(base_mlp, "act_fn")
        self.router = router
        self.hidden_size = int(self.gate_proj.in_features)
        self.intermediate_size = int(self.gate_proj.out_features)
        if motif_index.ndim != 1:
            raise ValueError("motif_index must be rank-1")
        if int(motif_index.numel()) != self.intermediate_size:
            raise ValueError("motif_index length must match intermediate_size")
        if int(motif_index.min().item()) < 0:
            raise ValueError("motif_index must be non-negative")
        self.num_motifs = int(motif_index.max().item()) + 1
        if self.num_motifs != int(self.router.num_motifs):
            raise ValueError("router num_motifs must match motif_index")
        self.register_buffer("motif_index", motif_index.to(dtype=torch.long), persistent=True)
        self._last_router_alpha: torch.Tensor | None = None
        if freeze_base:
            for param in self.gate_proj.parameters():
                param.requires_grad = False
            for param in self.up_proj.parameters():
                param.requires_grad = False
            for param in self.down_proj.parameters():
                param.requires_grad = False

    @staticmethod
    def _require_module(module: nn.Module, name: str) -> nn.Module:
        value = getattr(module, name, None)
        if not isinstance(value, nn.Module):
            raise ValueError(f"base_mlp must define nn.Module `{name}`")
        return value

    @staticmethod
    def _require_attr(module: nn.Module, name: str) -> Any:
        if not hasattr(module, name):
            raise ValueError(f"base_mlp must define `{name}`")
        return getattr(module, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate) * up  # [batch, seq, intermediate]

        router_input = x
        router_param = next(self.router.parameters(), None)
        if router_param is not None and router_input.dtype != router_param.dtype:
            router_input = router_input.to(dtype=router_param.dtype)
        alpha = self.router(router_input)
        if alpha.shape[:-1] != x.shape[:-1] or alpha.shape[-1] != self.num_motifs:
            raise ValueError("router output must have shape [batch, seq, num_motifs]")
        channel_scale = alpha[..., self.motif_index]  # [batch, seq, intermediate]
        if channel_scale.dtype != hidden.dtype:
            channel_scale = channel_scale.to(dtype=hidden.dtype)
        self._last_router_alpha = alpha.detach()
        return self.down_proj(hidden * channel_scale)

    def get_last_router_stats(self) -> dict[str, torch.Tensor]:
        if self._last_router_alpha is None:
            return {}
        alpha = self._last_router_alpha
        reduce_dims = tuple(range(alpha.ndim - 1))
        mean_alpha = alpha.mean(dim=reduce_dims)
        channel_counts = torch.bincount(self.motif_index, minlength=self.num_motifs).to(dtype=mean_alpha.dtype)
        return {
            "mean_alpha": mean_alpha,
            "channel_counts": channel_counts,
        }
