from __future__ import annotations

import torch

from src.model.qwen_motif_config import LowRankAdapterConfig, QwenFFNExpertLoRAConfig
from src.model.qwen_motif_lora import LowRankLinearAdapter, QwenMotifSplitLoRAMLP, RuntimeScaledLoRALinear
from src.model.qwen_motif_router import ContextualMotifRouter
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


def _build_tiny_qwen_mlp() -> Qwen2MLP:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    module = Qwen2MLP(config)
    module.float()
    return module


def test_low_rank_linear_adapter_is_zero_at_init() -> None:
    adapter = LowRankLinearAdapter(16, 12, LowRankAdapterConfig(rank=4, alpha=8.0))
    x = torch.randn(2, 3, 16)
    y = adapter(x)

    assert y.shape == (2, 3, 12)
    assert torch.allclose(y, torch.zeros_like(y))


def test_runtime_scaled_lora_linear_respects_scale() -> None:
    linear = torch.nn.Linear(8, 8, bias=False)
    wrapper = RuntimeScaledLoRALinear(linear, LowRankAdapterConfig(rank=2, alpha=4.0), freeze_base=True)
    x = torch.randn(2, 4, 8)
    wrapper.set_runtime_scale(torch.zeros(2, 4, 1))
    y0 = wrapper(x)
    wrapper.set_runtime_scale(torch.ones(2, 4, 1))
    y1 = wrapper(x)

    assert torch.allclose(y0, y1)


def test_low_rank_linear_adapter_handles_bfloat16_inputs() -> None:
    adapter = LowRankLinearAdapter(8, 8, LowRankAdapterConfig(rank=2, alpha=4.0))
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    y = adapter(x)

    assert y.dtype == torch.bfloat16


def test_qwen_motif_split_lora_mlp_preserves_init_and_trains_experts() -> None:
    torch.manual_seed(0)
    base_mlp = _build_tiny_qwen_mlp()
    motif_index = torch.arange(48, dtype=torch.long) % 3
    router = ContextualMotifRouter(hidden_size=32, num_motifs=3, router_hidden_size=12)
    module = QwenMotifSplitLoRAMLP(
        base_mlp=base_mlp,
        motif_index=motif_index,
        router=router,
        expert_configs={
            "expand": QwenFFNExpertLoRAConfig(
                up=LowRankAdapterConfig(rank=2, alpha=4.0),
                down=LowRankAdapterConfig(rank=2, alpha=4.0),
            ),
            "select": QwenFFNExpertLoRAConfig(gate=LowRankAdapterConfig(rank=2, alpha=4.0)),
            "memory": QwenFFNExpertLoRAConfig(
                gate=LowRankAdapterConfig(rank=2, alpha=4.0),
                up=LowRankAdapterConfig(rank=2, alpha=4.0),
                down=LowRankAdapterConfig(rank=2, alpha=4.0),
            ),
        },
        motif_names=("expand", "select", "memory"),
        freeze_base=True,
    )
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        y0 = module(x)
    loss = module(x).square().mean()
    loss.backward()

    assert y0.shape == x.shape
    assert torch.allclose(y0, base_mlp(x), atol=1e-6, rtol=1e-6)
    assert any(parameter.grad is not None for parameter in module.experts.parameters())
