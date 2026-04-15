from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.model.qwen_motif_config import QwenMotifAttentionPatchConfig
from src.model.qwen_motif_lora import RuntimeScaledLoRALinear
from src.model.qwen_motif_router import BaseMotifRouter


class QwenMotifAttentionAdapter(nn.Module):
    def __init__(
        self,
        base_attention: nn.Module,
        router: BaseMotifRouter,
        config: QwenMotifAttentionPatchConfig,
    ) -> None:
        super().__init__()
        self.base_attention = base_attention
        self.router = router
        self.config = config
        self.motif_names = tuple(config.motif_names)
        if len(self.motif_names) != 2:
            raise ValueError("attention motif adapter expects exactly two motifs")
        self.q_proj = RuntimeScaledLoRALinear(base_attention.q_proj, config.compare_q, freeze_base=config.freeze_base)
        self.k_proj = RuntimeScaledLoRALinear(base_attention.k_proj, config.compare_k, freeze_base=config.freeze_base)
        self.v_proj = RuntimeScaledLoRALinear(base_attention.v_proj, config.memory_v, freeze_base=config.freeze_base)
        self.o_proj = RuntimeScaledLoRALinear(base_attention.o_proj, config.memory_o, freeze_base=config.freeze_base)
        self.base_attention.q_proj = self.q_proj
        self.base_attention.k_proj = self.k_proj
        self.base_attention.v_proj = self.v_proj
        self.base_attention.o_proj = self.o_proj
        self._last_router_alpha: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        router_input = hidden_states
        router_param = next(self.router.parameters(), None)
        if router_param is not None and router_input.dtype != router_param.dtype:
            router_input = router_input.to(dtype=router_param.dtype)
        alpha = self.router(router_input)
        if alpha.shape[:-1] != hidden_states.shape[:-1] or alpha.shape[-1] != 2:
            raise ValueError("attention router output must have shape [batch, seq, 2]")
        compare_scale = alpha[..., 0:1]
        memory_scale = alpha[..., 1:2]
        self._last_router_alpha = alpha.detach()
        self.q_proj.set_runtime_scale(compare_scale)
        self.k_proj.set_runtime_scale(compare_scale)
        self.v_proj.set_runtime_scale(memory_scale)
        self.o_proj.set_runtime_scale(memory_scale)
        try:
            return self.base_attention(hidden_states, *args, **kwargs)
        finally:
            self.q_proj.clear_runtime_scale()
            self.k_proj.clear_runtime_scale()
            self.v_proj.clear_runtime_scale()
            self.o_proj.clear_runtime_scale()

    def get_last_router_stats(self) -> dict[str, torch.Tensor]:
        if self._last_router_alpha is None:
            return {}
        alpha = self._last_router_alpha
        reduce_dims = tuple(range(alpha.ndim - 1))
        return {"mean_alpha": alpha.mean(dim=reduce_dims)}

    def partial_reinit_(self, fraction: float = 1.0) -> None:
        self.q_proj.partial_reinit_(fraction=fraction)
        self.k_proj.partial_reinit_(fraction=fraction)
        self.v_proj.partial_reinit_(fraction=fraction)
        self.o_proj.partial_reinit_(fraction=fraction)
