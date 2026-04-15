from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.model.qwen_motif_config import QwenMotifPatchConfig, QwenMotifRouterConfig
from src.model.qwen_motif_ffn import QwenMotifSplitMLP
from src.model.qwen_motif_patch import (
    build_and_patch_qwen_ffn_layers,
    collect_qwen_motif_mlps,
    freeze_model_except_motif_routers,
    patch_qwen_ffn_layers,
)
from src.model.qwen_motif_router import StaticMotifRouter
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM



def _build_tiny_qwen_model() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config)
    model.float()
    model.eval()
    return model



def test_patch_qwen_ffn_layers_preserves_logits_at_init() -> None:
    torch.manual_seed(0)
    dense_model = _build_tiny_qwen_model()
    patched_model = copy.deepcopy(dense_model)
    motif_index = torch.arange(48, dtype=torch.long) % 3
    patch_qwen_ffn_layers(
        module=patched_model,
        layer_ids=(0, 1),
        motif_index_by_layer=motif_index,
        router_factory=lambda _layer_id: StaticMotifRouter(num_motifs=3),
        freeze_base=True,
    )
    input_ids = torch.randint(0, dense_model.config.vocab_size, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        dense_logits = dense_model(input_ids=input_ids, attention_mask=attention_mask).logits
        patched_logits = patched_model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(dense_logits, patched_logits, atol=1e-6, rtol=1e-6)



def test_build_and_patch_qwen_ffn_layers_patches_overlay_like_wrapper() -> None:
    class _Wrapper(nn.Module):
        def __init__(self, base_model: Qwen2ForCausalLM) -> None:
            super().__init__()
            self.base_model = base_model

    wrapper = _Wrapper(_build_tiny_qwen_model())
    config = QwenMotifPatchConfig(
        layer_ids=(0,),
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=12),
    )
    patched = build_and_patch_qwen_ffn_layers(wrapper, config)

    assert set(patched) == {0}
    assert isinstance(wrapper.base_model.model.layers[0].mlp, QwenMotifSplitMLP)



def test_freeze_model_except_motif_routers_leaves_only_router_trainable() -> None:
    model = _build_tiny_qwen_model()
    config = QwenMotifPatchConfig(
        layer_ids=(0, 1),
        freeze_model=True,
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=10),
    )
    build_and_patch_qwen_ffn_layers(model, config)
    freeze_model_except_motif_routers(model)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all("router" in name for name in trainable)



def test_collect_qwen_motif_mlps_returns_only_patched_layers() -> None:
    model = _build_tiny_qwen_model()
    build_and_patch_qwen_ffn_layers(
        model,
        QwenMotifPatchConfig(layer_ids=(1,)),
    )

    patched = collect_qwen_motif_mlps(model)

    assert set(patched) == {1}
    assert isinstance(patched[1], QwenMotifSplitMLP)
