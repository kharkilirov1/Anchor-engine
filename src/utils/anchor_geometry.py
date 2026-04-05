from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


_EPS = 1e-8
_LEADING_WHITESPACE_MARKERS = ("Ġ", "▁", "Ċ", "ĉ")


@dataclass(frozen=True)
class AnchorSpanMatch:
    anchor_text: str
    token_start: int
    token_end: int
    token_count: int
    char_start: int | None
    char_end: int | None
    match_method: str
    matched_text: str


def select_representative_layers(
    num_hidden_layers: int,
    count: int = 4,
) -> list[int]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    fractions = (0.25, 0.50, 0.75, 0.96)
    raw_layers = [
        max(1, min(num_hidden_layers, int(round(num_hidden_layers * fraction))))
        for fraction in fractions[:count]
    ]
    layers = sorted(set(raw_layers))
    if len(layers) >= count:
        return layers[:count]
    fallback = torch.linspace(1, num_hidden_layers, steps=min(num_hidden_layers, count)).round().int().tolist()
    for layer in fallback:
        layer_id = int(layer)
        if layer_id not in layers:
            layers.append(layer_id)
    return sorted(set(layers))


def list_model_layers(num_hidden_layers: int) -> list[int]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    return list(range(int(num_hidden_layers)))


def select_tail_probe_layers(
    num_hidden_layers: int,
    count: int = 10,
) -> list[int]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    width = min(int(count), int(num_hidden_layers))
    start = int(num_hidden_layers) - width
    return list(range(start, int(num_hidden_layers)))


def build_tail_reference_layers(
    probe_layers: list[int],
) -> dict[str, int]:
    if not probe_layers:
        raise ValueError("probe_layers must not be empty")
    layers = sorted(int(layer) for layer in probe_layers)
    mature_index = max(0, len(layers) - 4)
    template_prev_index = max(0, len(layers) - 2)
    template_curr_index = len(layers) - 1
    mature_layer = layers[mature_index]
    return {
        "slope_start_layer": layers[0],
        "slope_end_layer": mature_layer,
        "mature_layer": mature_layer,
        "template_prev_layer": layers[template_prev_index],
        "template_curr_layer": layers[template_curr_index],
    }


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _find_unique_substring(text: str, anchor_text: str) -> tuple[int, int] | None:
    matches = list(re.finditer(re.escape(anchor_text), text))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.start(), match.end()


def _decode_anchor_tokens(
    tokenizer: Any,
    token_ids: list[int],
) -> str:
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except Exception:
        return ""


def decode_token_surfaces(
    tokenizer: Any,
    token_ids: list[int],
) -> list[str]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        try:
            tokens = convert(token_ids)
            if isinstance(tokens, list) and len(tokens) == len(token_ids):
                return [str(token) for token in tokens]
        except Exception:
            pass
    return [_decode_anchor_tokens(tokenizer, [int(token_id)]) for token_id in token_ids]


def decode_token_pieces(
    tokenizer: Any,
    token_ids: list[int],
) -> list[str]:
    return [_decode_anchor_tokens(tokenizer, [int(token_id)]) for token_id in token_ids]


def token_has_leading_whitespace(
    raw_surface: str,
    decoded_piece: str,
) -> bool:
    if decoded_piece.startswith(" "):
        return True
    return raw_surface.startswith(_LEADING_WHITESPACE_MARKERS)


def _search_subsequence(
    full_ids: list[int],
    sub_ids: list[int],
) -> list[tuple[int, int]]:
    if not sub_ids or len(sub_ids) > len(full_ids):
        return []
    matches: list[tuple[int, int]] = []
    width = len(sub_ids)
    for start in range(len(full_ids) - width + 1):
        if full_ids[start : start + width] == sub_ids:
            matches.append((start, start + width - 1))
    return matches


def _match_from_offsets(
    *,
    text: str,
    anchor_text: str,
    offsets: list[tuple[int, int]],
) -> AnchorSpanMatch | None:
    char_span = _find_unique_substring(text, anchor_text)
    if char_span is None:
        return None
    char_start, char_end = char_span
    active_tokens = [
        idx
        for idx, (offset_start, offset_end) in enumerate(offsets)
        if offset_end > offset_start and offset_start < char_end and offset_end > char_start
    ]
    if not active_tokens:
        return None
    token_start = active_tokens[0]
    token_end = active_tokens[-1]
    matched_text = text[offsets[token_start][0] : offsets[token_end][1]]
    if _normalize_text(anchor_text) not in _normalize_text(matched_text):
        return None
    return AnchorSpanMatch(
        anchor_text=anchor_text,
        token_start=token_start,
        token_end=token_end,
        token_count=token_end - token_start + 1,
        char_start=char_start,
        char_end=char_end,
        match_method="offset_mapping",
        matched_text=matched_text,
    )


def _match_from_token_ids(
    *,
    anchor_text: str,
    input_ids: list[int],
    tokenizer: Any,
) -> AnchorSpanMatch | None:
    candidate_matches: dict[tuple[int, int], str] = {}
    for variant, label in ((anchor_text, "token_ids_exact"), (f" {anchor_text}", "token_ids_prefixed_space")):
        try:
            encoded = tokenizer(variant, add_special_tokens=False)
        except Exception:
            continue
        phrase_ids: Any
        if isinstance(encoded, Mapping):
            phrase_ids = encoded.get("input_ids")
        elif hasattr(encoded, "input_ids"):
            phrase_ids = getattr(encoded, "input_ids")
        elif hasattr(encoded, "__getitem__"):
            try:
                phrase_ids = encoded["input_ids"]
            except Exception:
                phrase_ids = encoded
        else:
            phrase_ids = encoded
        if phrase_ids is None:
            continue
        if isinstance(phrase_ids, torch.Tensor):
            phrase_seq = [int(token) for token in phrase_ids.reshape(-1).tolist()]
        else:
            phrase_seq = [int(token) for token in phrase_ids]
        for match in _search_subsequence(input_ids, phrase_seq):
            candidate_matches[match] = label
    if len(candidate_matches) != 1:
        return None
    (token_start, token_end), label = next(iter(candidate_matches.items()))
    matched_text = _decode_anchor_tokens(tokenizer, input_ids[token_start : token_end + 1])
    return AnchorSpanMatch(
        anchor_text=anchor_text,
        token_start=token_start,
        token_end=token_end,
        token_count=token_end - token_start + 1,
        char_start=None,
        char_end=None,
        match_method=label,
        matched_text=matched_text,
    )


def _match_from_decoded_pieces(
    *,
    anchor_text: str,
    input_ids: list[int],
    tokenizer: Any,
) -> AnchorSpanMatch | None:
    """Fallback: decode each token individually, build char→token map, search."""
    pieces = decode_token_pieces(tokenizer, input_ids)
    if not pieces:
        return None

    # Build cumulative decoded string with token boundary tracking
    char_to_token: list[int] = []
    decoded_full = ""
    for tok_idx, piece in enumerate(pieces):
        for _ in piece:
            char_to_token.append(tok_idx)
        decoded_full += piece

    if not decoded_full:
        return None

    # Search for anchor_text (case-insensitive to handle tokenizer quirks)
    needle = anchor_text.lower()
    haystack = decoded_full.lower()
    matches: list[int] = []
    start_pos = 0
    while True:
        pos = haystack.find(needle, start_pos)
        if pos == -1:
            break
        matches.append(pos)
        start_pos = pos + 1

    if len(matches) != 1:
        return None

    char_start = matches[0]
    char_end = char_start + len(needle)

    # Map character positions to token indices
    if char_start >= len(char_to_token) or char_end - 1 >= len(char_to_token):
        return None

    token_start = char_to_token[char_start]
    token_end = char_to_token[min(char_end - 1, len(char_to_token) - 1)]

    matched_text = decoded_full[char_start:char_end]
    return AnchorSpanMatch(
        anchor_text=anchor_text,
        token_start=token_start,
        token_end=token_end,
        token_count=token_end - token_start + 1,
        char_start=None,
        char_end=None,
        match_method="decoded_pieces",
        matched_text=matched_text,
    )


def match_anchor_span(
    *,
    text: str,
    anchor_text: str,
    input_ids: list[int],
    tokenizer: Any,
    offsets: list[tuple[int, int]] | None = None,
) -> AnchorSpanMatch | None:
    if offsets is not None:
        match = _match_from_offsets(
            text=text,
            anchor_text=anchor_text,
            offsets=offsets,
        )
        if match is not None:
            return match
    match = _match_from_token_ids(
        anchor_text=anchor_text,
        input_ids=input_ids,
        tokenizer=tokenizer,
    )
    if match is not None:
        return match
    return _match_from_decoded_pieces(
        anchor_text=anchor_text,
        input_ids=input_ids,
        tokenizer=tokenizer,
    )


def extract_delta_vectors(
    hidden_states: torch.Tensor,
    token_start: int,
    token_end: int,
) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be shaped [seq_len, hidden_dim]")
    if token_start < 0 or token_end < token_start or token_end >= hidden_states.size(0):
        raise ValueError("invalid token span")
    span_hidden = hidden_states[token_start : token_end + 1].to(dtype=torch.float32)
    if span_hidden.size(0) < 2:
        return span_hidden.new_zeros((0, span_hidden.size(-1)))
    return span_hidden[1:] - span_hidden[:-1]


def compute_geometry_metrics(
    delta_vectors: torch.Tensor,
) -> dict[str, float | int | None]:
    if delta_vectors.ndim != 2:
        raise ValueError("delta_vectors must be shaped [delta_count, hidden_dim]")
    delta_count = int(delta_vectors.size(0))
    token_count = delta_count + 1 if delta_count > 0 else 0
    if delta_count == 0:
        return {
            "token_count": token_count,
            "delta_count": delta_count,
            "mean_direction_norm": None,
            "mean_step_norm": None,
            "adjacent_cosine_coherence": None,
            "path_tortuosity": None,
            "rank1_explained_variance": None,
            "curvature_proxy": None,
        }

    deltas = delta_vectors.to(dtype=torch.float32)
    step_norms = deltas.norm(dim=-1)
    mean_direction = deltas.mean(dim=0)
    metrics: dict[str, float | int | None] = {
        "token_count": token_count,
        "delta_count": delta_count,
        "mean_direction_norm": float(mean_direction.norm().item()),
        "mean_step_norm": float(step_norms.mean().item()),
        "adjacent_cosine_coherence": None,
        "path_tortuosity": None,
        "rank1_explained_variance": None,
        "curvature_proxy": None,
    }
    if token_count < 4:
        return metrics

    adjacent: list[float] = []
    for idx in range(delta_count - 1):
        left = deltas[idx]
        right = deltas[idx + 1]
        left_norm = float(left.norm().item())
        right_norm = float(right.norm().item())
        if left_norm <= _EPS or right_norm <= _EPS:
            continue
        adjacent.append(float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item()))
    if adjacent:
        coherence = float(sum(adjacent) / len(adjacent))
        metrics["adjacent_cosine_coherence"] = coherence
        metrics["curvature_proxy"] = float(1.0 - coherence)

    displacement = deltas.sum(dim=0)
    displacement_norm = float(displacement.norm().item())
    if displacement_norm > _EPS:
        metrics["path_tortuosity"] = float(step_norms.sum().item() / displacement_norm)

    singular_values = torch.linalg.svdvals(deltas)
    energy = float((singular_values.square().sum()).item())
    if energy > _EPS:
        metrics["rank1_explained_variance"] = float((singular_values[0].item() ** 2) / energy)
    elif delta_count > 0:
        metrics["rank1_explained_variance"] = 1.0
    return metrics


def build_computability_flags(metrics: dict[str, float | int | None]) -> dict[str, bool]:
    return {
        "span_mean_direction": metrics.get("delta_count") is not None and int(metrics["delta_count"] or 0) > 0,
        "mean_direction_norm": metrics.get("mean_direction_norm") is not None,
        "mean_step_norm": metrics.get("mean_step_norm") is not None,
        "adjacent_cosine_coherence": metrics.get("adjacent_cosine_coherence") is not None,
        "path_tortuosity": metrics.get("path_tortuosity") is not None,
        "rank1_explained_variance": metrics.get("rank1_explained_variance") is not None,
        "curvature_proxy": metrics.get("curvature_proxy") is not None,
    }


def compute_mean_direction(
    delta_vectors: torch.Tensor,
) -> torch.Tensor | None:
    if delta_vectors.ndim != 2:
        raise ValueError("delta_vectors must be shaped [delta_count, hidden_dim]")
    if delta_vectors.size(0) == 0:
        return None
    return delta_vectors.to(dtype=torch.float32).mean(dim=0)


def compute_cross_prompt_stability(
    directions: list[torch.Tensor],
) -> dict[str, float | int | None]:
    valid = []
    for direction in directions:
        norm = float(direction.norm().item())
        if norm <= _EPS:
            continue
        valid.append(direction / norm)
    if len(valid) < 2:
        return {
            "pair_count": 0,
            "mean_pairwise_cosine": None,
            "std_pairwise_cosine": None,
            "min_pairwise_cosine": None,
            "max_pairwise_cosine": None,
        }
    values: list[float] = []
    for idx in range(len(valid)):
        for jdx in range(idx + 1, len(valid)):
            values.append(float(F.cosine_similarity(valid[idx].unsqueeze(0), valid[jdx].unsqueeze(0), dim=-1).item()))
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        "pair_count": len(values),
        "mean_pairwise_cosine": float(mean_value),
        "std_pairwise_cosine": float(math.sqrt(max(variance, 0.0))),
        "min_pairwise_cosine": float(min(values)),
        "max_pairwise_cosine": float(max(values)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auto-calibration: k-means clustering for mature/template/flat thresholds
# ─────────────────────────────────────────────────────────────────────────────

def auto_calibrate_thresholds(
    r1_references: list[float],
    delta_templates: list[float],
    n_clusters: int = 3,
    max_iter: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute cluster thresholds from observed r1/delta values via k-means.

    Returns dict with:
      - mature_r1_threshold: float
      - template_delta_threshold: float
      - cluster_centers: list of (r1_center, delta_center)
      - cluster_labels: list of "mature"/"template"/"flat" per sample
      - n_samples: int
    """
    n = len(r1_references)
    if n < 3 or len(delta_templates) != n:
        return {
            "mature_r1_threshold": 0.65,
            "template_delta_threshold": 0.08,
            "cluster_centers": [],
            "cluster_labels": [],
            "n_samples": n,
            "method": "fallback_default",
        }

    import random
    rng = random.Random(seed)

    # Normalize features to [0, 1] range for balanced clustering
    r1_arr = list(r1_references)
    dt_arr = list(delta_templates)
    r1_min, r1_max = min(r1_arr), max(r1_arr)
    dt_min, dt_max = min(dt_arr), max(dt_arr)
    r1_range = max(r1_max - r1_min, _EPS)
    dt_range = max(dt_max - dt_min, _EPS)

    points = [
        ((r - r1_min) / r1_range, (d - dt_min) / dt_range)
        for r, d in zip(r1_arr, dt_arr)
    ]

    # k-means initialization
    indices = rng.sample(range(n), min(n_clusters, n))
    centers = [points[i] for i in indices]

    for _ in range(max_iter):
        # Assign
        assignments: list[int] = []
        for p in points:
            dists = [(p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centers]
            assignments.append(int(min(range(len(dists)), key=lambda x: dists[x])))

        # Update
        new_centers: list[tuple[float, float]] = []
        for k in range(n_clusters):
            members = [points[i] for i in range(n) if assignments[i] == k]
            if not members:
                new_centers.append(centers[k])
            else:
                new_centers.append((
                    sum(m[0] for m in members) / len(members),
                    sum(m[1] for m in members) / len(members),
                ))
        if new_centers == centers:
            break
        centers = new_centers

    # Denormalize centers back to original scale
    real_centers = [
        (c[0] * r1_range + r1_min, c[1] * dt_range + dt_min)
        for c in centers
    ]

    # Identify clusters: highest r1 center = mature, highest delta center = template, rest = flat
    sorted_by_r1 = sorted(range(len(real_centers)), key=lambda i: real_centers[i][0], reverse=True)
    mature_idx = sorted_by_r1[0]
    remaining = [i for i in range(len(real_centers)) if i != mature_idx]
    template_idx = max(remaining, key=lambda i: real_centers[i][1])
    flat_idx = [i for i in remaining if i != template_idx][0] if len(remaining) > 1 else remaining[0]

    cluster_map = {mature_idx: "mature", template_idx: "template", flat_idx: "flat"}
    labels = [cluster_map.get(assignments[i], "flat") for i in range(n)]

    # Compute thresholds as midpoints between cluster centers
    mature_r1_center = real_centers[mature_idx][0]
    flat_r1_center = real_centers[flat_idx][0]
    mature_r1_threshold = (mature_r1_center + flat_r1_center) / 2.0

    template_delta_center = real_centers[template_idx][1]
    flat_delta_center = real_centers[flat_idx][1]
    template_delta_threshold = (template_delta_center + flat_delta_center) / 2.0

    return {
        "mature_r1_threshold": float(mature_r1_threshold),
        "template_delta_threshold": float(template_delta_threshold),
        "cluster_centers": [
            {"label": cluster_map[i], "r1": real_centers[i][0], "delta": real_centers[i][1]}
            for i in range(len(real_centers))
        ],
        "cluster_labels": labels,
        "n_samples": n,
        "method": "kmeans_3",
    }
