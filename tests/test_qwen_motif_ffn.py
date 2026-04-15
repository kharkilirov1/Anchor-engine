from __future__ import annotations

import torch

from src.model.qwen_motif_config import (
    build_contiguous_motif_index,
    build_layer_motif_indices,
)
from src.model.qwen_motif_ffn import QwenMotifSplitMLP
from src.model.qwen_motif_router import ContextualMotifRouter, StaticMotifRouter
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



def test_static_router_returns_unit_alpha_at_zero_init() -> None:
    router = StaticMotifRouter(num_motifs=3)
    hidden = torch.randn(2, 5, 7)
    alpha = router(hidden)

    assert alpha.shape == (2, 5, 3)
    assert torch.allclose(alpha, torch.ones_like(alpha))



def test_contextual_router_returns_unit_alpha_at_zero_init() -> None:
    router = ContextualMotifRouter(hidden_size=16, num_motifs=3, router_hidden_size=8)
    hidden = torch.randn(2, 5, 16)
    alpha = router(hidden)

    assert alpha.shape == (2, 5, 3)
    assert torch.allclose(alpha, torch.ones_like(alpha))



def test_qwen_motif_split_mlp_preserves_shape_and_backward() -> None:
    torch.manual_seed(0)
    base_mlp = _build_tiny_qwen_mlp()
    router = ContextualMotifRouter(hidden_size=32, num_motifs=3, router_hidden_size=12)
    motif_index = build_contiguous_motif_index(intermediate_size=48, num_motifs=3)
    module = QwenMotifSplitMLP(
        base_mlp=base_mlp,
        motif_index=motif_index,
        router=router,
        freeze_base=True,
    )
    x = torch.randn(2, 6, 32)
    y = module(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == x.shape
    assert module.router.out_proj.weight.grad is not None
    assert module.gate_proj.weight.grad is None
    stats = module.get_last_router_stats()
    assert torch.allclose(stats["mean_alpha"], torch.ones_like(stats["mean_alpha"]))



def test_build_layer_motif_indices_random_is_balanced() -> None:
    indices = build_layer_motif_indices(
        layer_ids=(2, 5),
        intermediate_size=48,
        num_motifs=3,
        assignment="random",
        seed=11,
    )

    assert set(indices) == {2, 5}
    counts = torch.bincount(indices[2], minlength=3)
    assert tuple(int(value) for value in counts.tolist()) == (16, 16, 16)
