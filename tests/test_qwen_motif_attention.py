from __future__ import annotations

import copy

import torch

from src.model.qwen_motif_attention import QwenMotifAttentionAdapter
from src.model.qwen_motif_config import LowRankAdapterConfig, QwenMotifAttentionPatchConfig, QwenMotifRouterConfig
from src.model.qwen_motif_patch import build_and_patch_qwen_attention_layers, collect_qwen_motif_attention_adapters
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


def test_attention_adapter_preserves_logits_at_init() -> None:
    torch.manual_seed(0)
    dense_model = _build_tiny_qwen_model()
    patched_model = copy.deepcopy(dense_model)
    config = QwenMotifAttentionPatchConfig(
        layer_ids=(0, 1),
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=10),
        compare_q=LowRankAdapterConfig(rank=2, alpha=4.0),
        compare_k=LowRankAdapterConfig(rank=2, alpha=4.0),
        memory_v=LowRankAdapterConfig(rank=2, alpha=4.0),
        memory_o=LowRankAdapterConfig(rank=2, alpha=4.0),
    )
    build_and_patch_qwen_attention_layers(patched_model, config)
    input_ids = torch.randint(0, dense_model.config.vocab_size, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        dense_logits = dense_model(input_ids=input_ids, attention_mask=attention_mask).logits
        patched_logits = patched_model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(dense_logits, patched_logits, atol=1e-6, rtol=1e-6)


def test_attention_adapter_collects_router_stats() -> None:
    model = _build_tiny_qwen_model()
    config = QwenMotifAttentionPatchConfig(
        layer_ids=(1,),
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=10),
        compare_q=LowRankAdapterConfig(rank=2, alpha=4.0),
        compare_k=LowRankAdapterConfig(rank=2, alpha=4.0),
        memory_v=LowRankAdapterConfig(rank=2, alpha=4.0),
        memory_o=LowRankAdapterConfig(rank=2, alpha=4.0),
    )
    build_and_patch_qwen_attention_layers(model, config)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    adapters = collect_qwen_motif_attention_adapters(model)

    assert set(adapters) == {1}
    assert isinstance(adapters[1], QwenMotifAttentionAdapter)
    stats = adapters[1].get_last_router_stats()
    assert torch.allclose(stats["mean_alpha"], torch.ones_like(stats["mean_alpha"]))
