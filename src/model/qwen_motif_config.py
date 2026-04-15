from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

DEFAULT_QWEN_MOTIFS: tuple[str, ...] = ("expand", "select", "memory")
DEFAULT_QWEN_ATTENTION_MOTIFS: tuple[str, ...] = ("compare", "memory")


@dataclass(frozen=True)
class QwenMotifRouterConfig:
    router_type: str = "static"
    temperature: float = 1.0
    hidden_size: int = 128
    hidden_multiplier: float = 0.5
    bias: bool = True
    top_k: int | None = None

    def resolve_hidden_size(self, model_hidden_size: int) -> int:
        if self.hidden_size > 0:
            return int(self.hidden_size)
        hidden = int(round(model_hidden_size * self.hidden_multiplier))
        return max(hidden, 1)


@dataclass(frozen=True)
class LowRankAdapterConfig:
    rank: int = 0
    alpha: float = 1.0
    dropout: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.rank > 0


@dataclass(frozen=True)
class QwenFFNExpertLoRAConfig:
    gate: LowRankAdapterConfig = field(default_factory=LowRankAdapterConfig)
    up: LowRankAdapterConfig = field(default_factory=LowRankAdapterConfig)
    down: LowRankAdapterConfig = field(default_factory=LowRankAdapterConfig)

    @property
    def enabled(self) -> bool:
        return self.gate.enabled or self.up.enabled or self.down.enabled


@dataclass(frozen=True)
class QwenMotifPatchConfig:
    layer_ids: tuple[int, ...]
    motif_names: tuple[str, ...] = DEFAULT_QWEN_MOTIFS
    motif_proportions: tuple[float, ...] | None = None
    assignment: str = "contiguous"
    random_seed: int = 0
    freeze_base: bool = True
    freeze_model: bool = False
    router: QwenMotifRouterConfig = field(default_factory=QwenMotifRouterConfig)
    expert_lora: Mapping[str, QwenFFNExpertLoRAConfig] | None = None


@dataclass(frozen=True)
class QwenMotifAttentionPatchConfig:
    layer_ids: tuple[int, ...]
    motif_names: tuple[str, ...] = DEFAULT_QWEN_ATTENTION_MOTIFS
    freeze_base: bool = True
    freeze_model: bool = False
    router: QwenMotifRouterConfig = field(default_factory=lambda: QwenMotifRouterConfig(router_type="contextual", hidden_size=128))
    compare_q: LowRankAdapterConfig = field(default_factory=LowRankAdapterConfig)
    compare_k: LowRankAdapterConfig = field(default_factory=LowRankAdapterConfig)
    memory_v: LowRankAdapterConfig = field(default_factory=LowRankAdapterConfig)
    memory_o: LowRankAdapterConfig = field(default_factory=LowRankAdapterConfig)


@dataclass(frozen=True)
class QwenMotifFullConfig:
    ffn: QwenMotifPatchConfig | None = None
    attention: QwenMotifAttentionPatchConfig | None = None


def build_layer_range(start: int, stop: int) -> tuple[int, ...]:
    if stop <= start:
        raise ValueError("stop must be greater than start")
    return tuple(range(int(start), int(stop)))


def build_default_ffn_lora_configs(
    motif_names: Sequence[str] = DEFAULT_QWEN_MOTIFS,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> dict[str, QwenFFNExpertLoRAConfig]:
    base = LowRankAdapterConfig(rank=rank, alpha=alpha, dropout=dropout)
    configs: dict[str, QwenFFNExpertLoRAConfig] = {}
    for motif_name in motif_names:
        name = str(motif_name).lower()
        if name == "expand":
            configs[motif_name] = QwenFFNExpertLoRAConfig(up=base, down=base)
        elif name == "select":
            configs[motif_name] = QwenFFNExpertLoRAConfig(gate=base)
        elif name == "memory":
            configs[motif_name] = QwenFFNExpertLoRAConfig(gate=base, up=base, down=base)
        else:
            configs[motif_name] = QwenFFNExpertLoRAConfig(gate=base, up=base, down=base)
    return configs


def build_default_attention_patch_config(
    layer_ids: tuple[int, ...],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    top_k: int | None = None,
) -> QwenMotifAttentionPatchConfig:
    adapter = LowRankAdapterConfig(rank=rank, alpha=alpha, dropout=dropout)
    return QwenMotifAttentionPatchConfig(
        layer_ids=layer_ids,
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=128, top_k=top_k),
        compare_q=adapter,
        compare_k=adapter,
        memory_v=adapter,
        memory_o=adapter,
    )


def _normalize_proportions(num_motifs: int, motif_proportions: Sequence[float] | None) -> tuple[float, ...]:
    if motif_proportions is None:
        return tuple(1.0 for _ in range(num_motifs))
    if len(motif_proportions) != num_motifs:
        raise ValueError("motif_proportions must match num_motifs")
    normalized = tuple(float(value) for value in motif_proportions)
    if any(value <= 0.0 for value in normalized):
        raise ValueError("motif_proportions must be positive")
    return normalized


def compute_motif_group_sizes(
    intermediate_size: int,
    num_motifs: int,
    motif_proportions: Sequence[float] | None = None,
) -> tuple[int, ...]:
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if num_motifs <= 0:
        raise ValueError("num_motifs must be positive")
    proportions = _normalize_proportions(num_motifs, motif_proportions)
    total = sum(proportions)
    raw_sizes = [intermediate_size * value / total for value in proportions]
    base_sizes = [int(size) for size in raw_sizes]
    remainder = int(intermediate_size - sum(base_sizes))
    ranked = sorted(
        range(num_motifs),
        key=lambda index: (raw_sizes[index] - base_sizes[index], -index),
        reverse=True,
    )
    for index in ranked[:remainder]:
        base_sizes[index] += 1
    if any(size <= 0 for size in base_sizes):
        raise ValueError("each motif must receive at least one channel")
    return tuple(base_sizes)


def build_contiguous_motif_index(
    intermediate_size: int,
    num_motifs: int,
    motif_proportions: Sequence[float] | None = None,
) -> torch.Tensor:
    group_sizes = compute_motif_group_sizes(
        intermediate_size=intermediate_size,
        num_motifs=num_motifs,
        motif_proportions=motif_proportions,
    )
    motif_ids = [torch.full((size,), motif_id, dtype=torch.long) for motif_id, size in enumerate(group_sizes)]
    return torch.cat(motif_ids, dim=0)


def build_random_motif_index(
    intermediate_size: int,
    num_motifs: int,
    motif_proportions: Sequence[float] | None = None,
    seed: int = 0,
) -> torch.Tensor:
    motif_index = build_contiguous_motif_index(
        intermediate_size=intermediate_size,
        num_motifs=num_motifs,
        motif_proportions=motif_proportions,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(intermediate_size, generator=generator)
    return motif_index[permutation]


def build_round_robin_motif_index(intermediate_size: int, num_motifs: int) -> torch.Tensor:
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if num_motifs <= 0:
        raise ValueError("num_motifs must be positive")
    return torch.arange(intermediate_size, dtype=torch.long) % num_motifs


def build_motif_index(
    intermediate_size: int,
    num_motifs: int,
    assignment: str = "contiguous",
    motif_proportions: Sequence[float] | None = None,
    seed: int = 0,
) -> torch.Tensor:
    assignment_name = str(assignment).lower()
    if assignment_name == "contiguous":
        return build_contiguous_motif_index(
            intermediate_size=intermediate_size,
            num_motifs=num_motifs,
            motif_proportions=motif_proportions,
        )
    if assignment_name == "random":
        return build_random_motif_index(
            intermediate_size=intermediate_size,
            num_motifs=num_motifs,
            motif_proportions=motif_proportions,
            seed=seed,
        )
    if assignment_name in {"round_robin", "interleaved"}:
        return build_round_robin_motif_index(intermediate_size=intermediate_size, num_motifs=num_motifs)
    raise ValueError(f"unsupported assignment: {assignment}")


def build_layer_motif_indices(
    layer_ids: Sequence[int],
    intermediate_size: int,
    num_motifs: int,
    assignment: str = "contiguous",
    motif_proportions: Sequence[float] | None = None,
    seed: int = 0,
) -> dict[int, torch.Tensor]:
    motif_indices: dict[int, torch.Tensor] = {}
    for layer_id in layer_ids:
        layer_seed = int(seed) + int(layer_id)
        motif_indices[int(layer_id)] = build_motif_index(
            intermediate_size=intermediate_size,
            num_motifs=num_motifs,
            assignment=assignment,
            motif_proportions=motif_proportions,
            seed=layer_seed,
        )
    return motif_indices
