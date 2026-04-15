from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from src.model.qwen_motif_config import LowRankAdapterConfig, QwenFFNExpertLoRAConfig
from src.model.qwen_motif_ffn import QwenMotifSplitMLP
from src.model.qwen_motif_router import BaseMotifRouter


class LowRankLinearAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LowRankAdapterConfig,
    ) -> None:
        super().__init__()
        if config.rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(config.rank)
        self.alpha = float(config.alpha)
        self.scale = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.down = nn.Linear(self.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter_input = x if x.dtype == self.down.weight.dtype else x.to(dtype=self.down.weight.dtype)
        delta = self.up(self.dropout(self.down(adapter_input))) * self.scale
        if delta.dtype != x.dtype:
            delta = delta.to(dtype=x.dtype)
        return delta

    def partial_reinit_(self, fraction: float = 1.0) -> None:
        if fraction <= 0.0:
            return
        if fraction >= 1.0:
            self.reset_parameters()
            return
        with torch.no_grad():
            down_mask = torch.rand_like(self.down.weight) < fraction
            up_mask = torch.rand_like(self.up.weight) < fraction
            down_noise = torch.empty_like(self.down.weight)
            nn.init.kaiming_uniform_(down_noise, a=math.sqrt(5.0))
            self.down.weight.copy_(torch.where(down_mask, down_noise, self.down.weight))
            self.up.weight.copy_(torch.where(up_mask, torch.zeros_like(self.up.weight), self.up.weight))


class RuntimeScaledLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        config: LowRankAdapterConfig,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.adapter = LowRankLinearAdapter(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            config=config,
        ) if config.enabled else None
        self._runtime_scale: torch.Tensor | float | None = None
        if freeze_base:
            for param in self.base_linear.parameters():
                param.requires_grad = False

    def set_runtime_scale(self, scale: torch.Tensor | float | None) -> None:
        self._runtime_scale = scale

    def clear_runtime_scale(self) -> None:
        self._runtime_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_linear(x)
        if self.adapter is None:
            return out
        delta = self.adapter(x)
        scale = self._runtime_scale
        if scale is None:
            return out + delta
        if torch.is_tensor(scale):
            scale_tensor = scale
            while scale_tensor.ndim < delta.ndim:
                scale_tensor = scale_tensor.unsqueeze(-1)
            if scale_tensor.dtype != delta.dtype:
                scale_tensor = scale_tensor.to(dtype=delta.dtype)
            return out + delta * scale_tensor
        return out + delta * float(scale)

    def partial_reinit_(self, fraction: float = 1.0) -> None:
        if self.adapter is not None:
            self.adapter.partial_reinit_(fraction=fraction)


class FFNMotifLoRAExpert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        motif_mask: torch.Tensor,
        base_down_proj: nn.Module,
        act_fn,
        config: QwenFFNExpertLoRAConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.base_down_proj_forward = base_down_proj.forward
        self.act_fn = act_fn
        self.gate_adapter = LowRankLinearAdapter(self.hidden_size, self.intermediate_size, config.gate) if config.gate.enabled else None
        self.up_adapter = LowRankLinearAdapter(self.hidden_size, self.intermediate_size, config.up) if config.up.enabled else None
        self.down_adapter = LowRankLinearAdapter(self.intermediate_size, self.hidden_size, config.down) if config.down.enabled else None
        self.has_input_adapters = self.gate_adapter is not None or self.up_adapter is not None
        mask = motif_mask.to(dtype=torch.float32).view(1, 1, self.intermediate_size)
        self.register_buffer("motif_mask", mask, persistent=True)

    def forward(
        self,
        x: torch.Tensor,
        gate_base: torch.Tensor,
        up_base: torch.Tensor,
        hidden_base: torch.Tensor,
    ) -> torch.Tensor:
        adapted_hidden = hidden_base
        if self.has_input_adapters:
            gate = gate_base
            up = up_base
            if self.gate_adapter is not None:
                gate = gate + self.gate_adapter(x)
            if self.up_adapter is not None:
                up = up + self.up_adapter(x)
            adapted_hidden = self.act_fn(gate) * up
        hidden_delta = (adapted_hidden - hidden_base) * self.motif_mask
        output_delta = self.base_down_proj_forward(hidden_delta) if self.has_input_adapters else 0.0
        motif_hidden = adapted_hidden * self.motif_mask
        if self.down_adapter is not None:
            output_delta = output_delta + self.down_adapter(motif_hidden)
        return output_delta

    def partial_reinit_(self, fraction: float = 1.0) -> None:
        for adapter in (self.gate_adapter, self.up_adapter, self.down_adapter):
            if adapter is not None:
                adapter.partial_reinit_(fraction=fraction)


class QwenMotifSplitLoRAMLP(QwenMotifSplitMLP):
    def __init__(
        self,
        base_mlp: nn.Module,
        motif_index: torch.Tensor,
        router: BaseMotifRouter,
        expert_configs: Mapping[int | str, QwenFFNExpertLoRAConfig],
        motif_names: tuple[str, ...] | None = None,
        freeze_base: bool = True,
    ) -> None:
        super().__init__(
            base_mlp=base_mlp,
            motif_index=motif_index,
            router=router,
            freeze_base=freeze_base,
        )
        names = motif_names or tuple(str(index) for index in range(self.num_motifs))
        if len(names) != self.num_motifs:
            raise ValueError("motif_names length must match num_motifs")
        self.motif_names = tuple(names)
        experts: dict[str, FFNMotifLoRAExpert] = {}
        for motif_id, motif_name in enumerate(self.motif_names):
            config = expert_configs.get(motif_id) or expert_configs.get(motif_name)
            if config is None or not config.enabled:
                continue
            motif_mask = self.motif_index == motif_id
            experts[str(motif_id)] = FFNMotifLoRAExpert(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                motif_mask=motif_mask,
                base_down_proj=self.down_proj,
                act_fn=self.act_fn,
                config=config,
            )
        self.experts = nn.ModuleDict(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate) * up

        router_input = x
        router_param = next(self.router.parameters(), None)
        if router_param is not None and router_input.dtype != router_param.dtype:
            router_input = router_input.to(dtype=router_param.dtype)
        alpha = self.router(router_input)
        if alpha.shape[:-1] != x.shape[:-1] or alpha.shape[-1] != self.num_motifs:
            raise ValueError("router output must have shape [batch, seq, num_motifs]")
        channel_scale = alpha[..., self.motif_index]
        if channel_scale.dtype != hidden.dtype:
            channel_scale = channel_scale.to(dtype=hidden.dtype)
        self._last_router_alpha = alpha.detach()
        out = self.down_proj(hidden * channel_scale)
        for motif_key, expert in self.experts.items():
            motif_id = int(motif_key)
            beta = alpha[..., motif_id:motif_id + 1]
            if beta.dtype != hidden.dtype:
                beta = beta.to(dtype=hidden.dtype)
            out = out + beta * expert(x=x, gate_base=gate, up_base=up, hidden_base=hidden)
        return out

    def partial_reinit_(self, fraction: float = 1.0) -> None:
        for expert in self.experts.values():
            expert.partial_reinit_(fraction=fraction)
