from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from src.model.qwen_motif_attention import QwenMotifAttentionAdapter
from src.model.qwen_motif_config import (
    QwenMotifAttentionPatchConfig,
    QwenMotifFullConfig,
    QwenMotifPatchConfig,
    build_layer_motif_indices,
)
from src.model.qwen_motif_ffn import QwenMotifSplitMLP
from src.model.qwen_motif_lora import QwenMotifSplitLoRAMLP
from src.model.qwen_motif_router import BaseMotifRouter, build_motif_router

QWEN_MOTIF_MLP_TYPES = (QwenMotifSplitMLP, QwenMotifSplitLoRAMLP)


def get_qwen_decoder_layers(module: nn.Module) -> nn.ModuleList:
    if hasattr(module, "model") and hasattr(module.model, "layers") and isinstance(module.model.layers, nn.ModuleList):
        return module.model.layers
    base_model = getattr(module, "base_model", None)
    if isinstance(base_model, nn.Module):
        return get_qwen_decoder_layers(base_model)
    raise ValueError("could not resolve Qwen decoder layers from module")


def patch_qwen_ffn_layers(
    module: nn.Module,
    layer_ids: Sequence[int],
    motif_index_by_layer: Mapping[int, torch.Tensor] | torch.Tensor,
    router_factory: Callable[[int], BaseMotifRouter],
    freeze_base: bool = True,
) -> dict[int, QwenMotifSplitMLP]:
    layers = get_qwen_decoder_layers(module)
    patched_layers: dict[int, QwenMotifSplitMLP] = {}
    for layer_id in layer_ids:
        index = int(layer_id)
        if index < 0 or index >= len(layers):
            raise IndexError(f"layer_id {index} is out of range")
        layer = layers[index]
        base_mlp = layer.mlp
        if isinstance(base_mlp, QWEN_MOTIF_MLP_TYPES):
            raise ValueError(f"layer {index} is already patched with a Qwen motif MLP")
        motif_index = motif_index_by_layer if isinstance(motif_index_by_layer, torch.Tensor) else motif_index_by_layer[index]
        patched = QwenMotifSplitMLP(
            base_mlp=base_mlp,
            motif_index=motif_index.detach().clone(),
            router=router_factory(index),
            freeze_base=freeze_base,
        )
        layer.mlp = patched
        patched_layers[index] = patched
    return patched_layers


def build_and_patch_qwen_ffn_lora_layers(
    module: nn.Module,
    config: QwenMotifPatchConfig,
    motif_index_by_layer: Mapping[int, torch.Tensor] | None = None,
    hidden_size: int | None = None,
) -> dict[int, QwenMotifSplitLoRAMLP]:
    layers = get_qwen_decoder_layers(module)
    if not config.layer_ids:
        return {}
    first_layer = layers[int(config.layer_ids[0])]
    model_hidden_size = int(first_layer.mlp.gate_proj.in_features) if hidden_size is None else int(hidden_size)
    intermediate_size = int(first_layer.mlp.gate_proj.out_features)
    layer_motif_indices = motif_index_by_layer or build_layer_motif_indices(
        layer_ids=config.layer_ids,
        intermediate_size=intermediate_size,
        num_motifs=len(config.motif_names),
        assignment=config.assignment,
        motif_proportions=config.motif_proportions,
        seed=config.random_seed,
    )
    patched_layers: dict[int, QwenMotifSplitLoRAMLP] = {}
    for layer_id in config.layer_ids:
        index = int(layer_id)
        layer = layers[index]
        base_mlp = layer.mlp
        if isinstance(base_mlp, QWEN_MOTIF_MLP_TYPES):
            raise ValueError(f"layer {index} is already patched with a Qwen motif MLP")
        router = build_motif_router(
            config=config.router,
            model_hidden_size=model_hidden_size,
            num_motifs=len(config.motif_names),
        ).module
        patched = QwenMotifSplitLoRAMLP(
            base_mlp=base_mlp,
            motif_index=layer_motif_indices[index].detach().clone(),
            router=router,
            expert_configs=config.expert_lora or {},
            motif_names=config.motif_names,
            freeze_base=config.freeze_base,
        )
        layer.mlp = patched
        patched_layers[index] = patched
    if config.freeze_model:
        freeze_model_except_qwen_motif_trainables(module)
    return patched_layers


def build_and_patch_qwen_ffn_layers(
    module: nn.Module,
    config: QwenMotifPatchConfig,
) -> dict[int, nn.Module]:
    layers = get_qwen_decoder_layers(module)
    if not config.layer_ids:
        return {}
    first_layer = layers[int(config.layer_ids[0])]
    hidden_size = int(first_layer.mlp.gate_proj.in_features)
    intermediate_size = int(first_layer.mlp.gate_proj.out_features)
    motif_index_by_layer = build_layer_motif_indices(
        layer_ids=config.layer_ids,
        intermediate_size=intermediate_size,
        num_motifs=len(config.motif_names),
        assignment=config.assignment,
        motif_proportions=config.motif_proportions,
        seed=config.random_seed,
    )

    if config.expert_lora:
        return build_and_patch_qwen_ffn_lora_layers(
            module=module,
            config=config,
            motif_index_by_layer=motif_index_by_layer,
            hidden_size=hidden_size,
        )

    def router_factory(_layer_id: int) -> BaseMotifRouter:
        return build_motif_router(
            config=config.router,
            model_hidden_size=hidden_size,
            num_motifs=len(config.motif_names),
        ).module

    patched_layers = patch_qwen_ffn_layers(
        module=module,
        layer_ids=config.layer_ids,
        motif_index_by_layer=motif_index_by_layer,
        router_factory=router_factory,
        freeze_base=config.freeze_base,
    )
    if config.freeze_model:
        freeze_model_except_qwen_motif_trainables(module)
    return patched_layers


def collect_qwen_motif_mlps(module: nn.Module) -> dict[int, nn.Module]:
    layers = get_qwen_decoder_layers(module)
    collected: dict[int, nn.Module] = {}
    for layer_id, layer in enumerate(layers):
        if isinstance(layer.mlp, QWEN_MOTIF_MLP_TYPES):
            collected[int(layer_id)] = layer.mlp
    return collected


def patch_qwen_attention_layers(
    module: nn.Module,
    layer_ids: Sequence[int],
    router_factory: Callable[[int], BaseMotifRouter],
    config: QwenMotifAttentionPatchConfig,
) -> dict[int, QwenMotifAttentionAdapter]:
    layers = get_qwen_decoder_layers(module)
    patched_layers: dict[int, QwenMotifAttentionAdapter] = {}
    for layer_id in layer_ids:
        index = int(layer_id)
        if index < 0 or index >= len(layers):
            raise IndexError(f"layer_id {index} is out of range")
        layer = layers[index]
        base_attention = layer.self_attn
        if isinstance(base_attention, QwenMotifAttentionAdapter):
            raise ValueError(f"layer {index} is already patched with QwenMotifAttentionAdapter")
        patched = QwenMotifAttentionAdapter(
            base_attention=base_attention,
            router=router_factory(index),
            config=config,
        )
        layer.self_attn = patched
        patched_layers[index] = patched
    return patched_layers


def build_and_patch_qwen_attention_layers(
    module: nn.Module,
    config: QwenMotifAttentionPatchConfig,
) -> dict[int, QwenMotifAttentionAdapter]:
    layers = get_qwen_decoder_layers(module)
    if not config.layer_ids:
        return {}
    first_layer = layers[int(config.layer_ids[0])]
    hidden_size = int(first_layer.self_attn.q_proj.in_features)

    def router_factory(_layer_id: int) -> BaseMotifRouter:
        return build_motif_router(
            config=config.router,
            model_hidden_size=hidden_size,
            num_motifs=len(config.motif_names),
        ).module

    patched_layers = patch_qwen_attention_layers(
        module=module,
        layer_ids=config.layer_ids,
        router_factory=router_factory,
        config=config,
    )
    if config.freeze_model:
        freeze_model_except_qwen_motif_trainables(module)
    return patched_layers


def collect_qwen_motif_attention_adapters(module: nn.Module) -> dict[int, QwenMotifAttentionAdapter]:
    layers = get_qwen_decoder_layers(module)
    collected: dict[int, QwenMotifAttentionAdapter] = {}
    for layer_id, layer in enumerate(layers):
        if isinstance(layer.self_attn, QwenMotifAttentionAdapter):
            collected[int(layer_id)] = layer.self_attn
    return collected


def freeze_model_except_motif_routers(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    for motif_mlp in collect_qwen_motif_mlps(module).values():
        for parameter in motif_mlp.router.parameters():
            parameter.requires_grad = True
    for attn_adapter in collect_qwen_motif_attention_adapters(module).values():
        for parameter in attn_adapter.router.parameters():
            parameter.requires_grad = True


def freeze_model_except_qwen_motif_trainables(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    for motif_mlp in collect_qwen_motif_mlps(module).values():
        for parameter in motif_mlp.router.parameters():
            parameter.requires_grad = True
        for expert in getattr(motif_mlp, "experts", {}).values():
            for parameter in expert.parameters():
                parameter.requires_grad = True
    for attn_adapter in collect_qwen_motif_attention_adapters(module).values():
        for parameter in attn_adapter.router.parameters():
            parameter.requires_grad = True
        for projection in (attn_adapter.q_proj, attn_adapter.k_proj, attn_adapter.v_proj, attn_adapter.o_proj):
            if getattr(projection, "adapter", None) is not None:
                for parameter in projection.adapter.parameters():
                    parameter.requires_grad = True


def collect_qwen_motif_trainable_names(module: nn.Module) -> list[str]:
    return [name for name, parameter in module.named_parameters() if parameter.requires_grad]


def partial_reinit_qwen_motif_modules(module: nn.Module, fraction: float = 1.0) -> None:
    for motif_mlp in collect_qwen_motif_mlps(module).values():
        if hasattr(motif_mlp, "partial_reinit_"):
            motif_mlp.partial_reinit_(fraction=fraction)
    for attn_adapter in collect_qwen_motif_attention_adapters(module).values():
        attn_adapter.partial_reinit_(fraction=fraction)


def apply_qwen_motif_pipeline(module: nn.Module, config: QwenMotifFullConfig) -> dict[str, dict[int, nn.Module]]:
    results: dict[str, dict[int, nn.Module]] = {}
    if config.ffn is not None:
        results["ffn"] = build_and_patch_qwen_ffn_layers(module, config.ffn)
    if config.attention is not None:
        results["attention"] = build_and_patch_qwen_attention_layers(module, config.attention)
    return results
