from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.qwen_motif_config import QwenMotifRouterConfig


class BaseMotifRouter(nn.Module):
    def __init__(self, num_motifs: int, temperature: float = 1.0, top_k: int | None = None) -> None:
        super().__init__()
        if num_motifs <= 0:
            raise ValueError("num_motifs must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and (top_k <= 0 or top_k > num_motifs):
            raise ValueError("top_k must be in [1, num_motifs]")
        self.num_motifs = int(num_motifs)
        self.temperature = float(temperature)
        self.top_k = None if top_k is None else int(top_k)

    def _normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.temperature, dim=-1)
        if self.top_k is not None and self.top_k < self.num_motifs:
            top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
            sparse_probs = torch.zeros_like(probs)
            sparse_probs.scatter_(-1, top_indices, top_values)
            probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return float(self.num_motifs) * probs


class StaticMotifRouter(BaseMotifRouter):
    def __init__(
        self,
        num_motifs: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        init_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__(num_motifs=num_motifs, temperature=temperature, top_k=top_k)
        initial = torch.zeros(self.num_motifs, dtype=torch.float32)
        if init_logits is not None:
            if init_logits.shape != initial.shape:
                raise ValueError("init_logits must have shape [num_motifs]")
            initial.copy_(init_logits.to(dtype=initial.dtype))
        self.logits = nn.Parameter(initial)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_shape = hidden_states.shape[:-1]
        logits = self.logits.view(*([1] * len(batch_shape)), self.num_motifs)
        alpha = self._normalize_logits(logits)
        return alpha.expand(*batch_shape, self.num_motifs)


class ContextualMotifRouter(BaseMotifRouter):
    def __init__(
        self,
        hidden_size: int,
        num_motifs: int,
        router_hidden_size: int,
        temperature: float = 1.0,
        bias: bool = True,
        top_k: int | None = None,
    ) -> None:
        super().__init__(num_motifs=num_motifs, temperature=temperature, top_k=top_k)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if router_hidden_size <= 0:
            raise ValueError("router_hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.router_hidden_size = int(router_hidden_size)
        self.in_proj = nn.Linear(self.hidden_size, self.router_hidden_size, bias=bias)
        self.out_proj = nn.Linear(self.router_hidden_size, self.num_motifs, bias=bias)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.out_proj(F.silu(self.in_proj(hidden_states)))
        return self._normalize_logits(logits)


@dataclass(frozen=True)
class BuiltRouter:
    name: str
    module: BaseMotifRouter


def build_motif_router(
    config: QwenMotifRouterConfig,
    model_hidden_size: int,
    num_motifs: int,
) -> BuiltRouter:
    router_type = str(config.router_type).lower()
    if router_type == "static":
        router = StaticMotifRouter(
            num_motifs=num_motifs,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        return BuiltRouter(name="static", module=router)
    if router_type == "contextual":
        router_hidden_size = config.resolve_hidden_size(model_hidden_size)
        router = ContextualMotifRouter(
            hidden_size=model_hidden_size,
            num_motifs=num_motifs,
            router_hidden_size=router_hidden_size,
            temperature=config.temperature,
            bias=config.bias,
            top_k=config.top_k,
        )
        return BuiltRouter(name="contextual", module=router)
    raise ValueError(f"unsupported router_type: {config.router_type}")
