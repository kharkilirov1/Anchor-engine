from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from statistics import mean, median

import torch


@dataclass(frozen=True)
class TokenStructureStats:
    token_count: int
    vocab_size: int
    unique_tokens: int
    unique_token_ratio: float
    unigram_entropy_bits: float
    top_8_mass: float
    repeat_within_4: float
    repeat_within_8: float
    repeat_within_16: float
    mean_repeat_distance: float
    median_repeat_distance: float
    conditional_entropy_bits: float
    mean_next_token_peak_prob: float


@dataclass(frozen=True)
class SequenceStructureStats:
    num_sequences: int
    seq_len: int
    unique_sequences: int
    sequence_uniqueness_ratio: float
    top_sequence_mass: float


def _entropy_from_counts(counts: torch.Tensor) -> float:
    probs = counts.float() / counts.sum().clamp_min(1)
    probs = probs[probs > 0]
    if probs.numel() == 0:
        return 0.0
    return float((-(probs * probs.log2())).sum().item())


def _repeat_stats(token_ids: torch.Tensor, window: int) -> float:
    ids = token_ids.tolist()
    hits = 0
    total = 0
    for idx, token in enumerate(ids):
        if idx == 0:
            continue
        total += 1
        start = max(0, idx - window)
        if token in ids[start:idx]:
            hits += 1
    return hits / max(total, 1)


def _repeat_distances(token_ids: torch.Tensor) -> tuple[float, float]:
    last_seen: dict[int, int] = {}
    distances: list[int] = []
    for idx, token in enumerate(token_ids.tolist()):
        if token in last_seen:
            distances.append(idx - last_seen[token])
        last_seen[token] = idx
    if not distances:
        return 0.0, 0.0
    return float(mean(distances)), float(median(distances))


def _conditional_stats(token_ids: torch.Tensor, vocab_size: int) -> tuple[float, float]:
    if token_ids.numel() < 2:
        return 0.0, 0.0
    pair_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.long)
    src = token_ids[:-1].long()
    dst = token_ids[1:].long()
    for a, b in zip(src.tolist(), dst.tolist()):
        pair_counts[a, b] += 1

    totals = pair_counts.sum(dim=1)
    entropies: list[float] = []
    peaks: list[float] = []
    weights: list[float] = []
    for row, total in zip(pair_counts, totals):
        total_int = int(total.item())
        if total_int <= 0:
            continue
        probs = row.float() / total_int
        nz = probs[probs > 0]
        entropies.append(float((-(nz * nz.log2())).sum().item()))
        peaks.append(float(probs.max().item()))
        weights.append(total_int)
    if not weights:
        return 0.0, 0.0
    total_weight = sum(weights)
    weighted_entropy = sum(e * w for e, w in zip(entropies, weights)) / total_weight
    weighted_peak = sum(p * w for p, w in zip(peaks, weights)) / total_weight
    return weighted_entropy, weighted_peak


def compute_token_structure_stats(token_ids: torch.Tensor, vocab_size: int | None = None) -> TokenStructureStats:
    flat = token_ids.detach().flatten().to(torch.long).cpu()
    if flat.numel() == 0:
        raise ValueError("token_ids must be non-empty")
    if vocab_size is None:
        vocab_size = int(flat.max().item()) + 1
    counts = torch.bincount(flat, minlength=vocab_size)
    sorted_counts, _ = torch.sort(counts, descending=True)
    top_8_mass = float(sorted_counts[:8].sum().item() / flat.numel())
    mean_repeat_distance, median_repeat_distance = _repeat_distances(flat)
    conditional_entropy_bits, mean_next_token_peak_prob = _conditional_stats(flat, vocab_size)
    unique_tokens = int((counts > 0).sum().item())
    return TokenStructureStats(
        token_count=int(flat.numel()),
        vocab_size=int(vocab_size),
        unique_tokens=unique_tokens,
        unique_token_ratio=unique_tokens / max(int(flat.numel()), 1),
        unigram_entropy_bits=_entropy_from_counts(counts),
        top_8_mass=top_8_mass,
        repeat_within_4=_repeat_stats(flat, 4),
        repeat_within_8=_repeat_stats(flat, 8),
        repeat_within_16=_repeat_stats(flat, 16),
        mean_repeat_distance=mean_repeat_distance,
        median_repeat_distance=median_repeat_distance,
        conditional_entropy_bits=conditional_entropy_bits,
        mean_next_token_peak_prob=mean_next_token_peak_prob,
    )


def compute_sequence_structure_stats(sequences: torch.Tensor) -> SequenceStructureStats:
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape [N, T]")
    rows = [tuple(int(v) for v in row.tolist()) for row in sequences.cpu()]
    counts: dict[tuple[int, ...], int] = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    top_sequence_mass = max(counts.values()) / max(len(rows), 1)
    return SequenceStructureStats(
        num_sequences=len(rows),
        seq_len=int(sequences.size(1)),
        unique_sequences=len(counts),
        sequence_uniqueness_ratio=len(counts) / max(len(rows), 1),
        top_sequence_mass=top_sequence_mass,
    )


def stats_to_dict(stats: TokenStructureStats | SequenceStructureStats) -> dict[str, float | int]:
    return asdict(stats)
