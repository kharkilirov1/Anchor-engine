from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.utils.anchor_geometry import (
    decode_token_pieces,
    decode_token_surfaces,
    match_anchor_span,
)


_EPS = 1e-8


@dataclass(frozen=True)
class SpanEncoding:
    text: str
    focus_text: str
    input_ids: list[int]
    attention_mask: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...]
    span_match: Any
    raw_tokens: list[str]
    decoded_tokens: list[str]


def _tensor_offsets_to_list(offset_mapping: Any) -> list[tuple[int, int]] | None:
    if offset_mapping is None:
        return None
    if isinstance(offset_mapping, torch.Tensor):
        pairs = offset_mapping.squeeze(0).tolist()
        return [(int(start), int(end)) for start, end in pairs]
    if isinstance(offset_mapping, list):
        if offset_mapping and isinstance(offset_mapping[0], tuple):
            return [(int(start), int(end)) for start, end in offset_mapping]
        if offset_mapping and isinstance(offset_mapping[0], list):
            if offset_mapping and offset_mapping[0] and isinstance(offset_mapping[0][0], (list, tuple)):
                return [(int(start), int(end)) for start, end in offset_mapping[0]]
            return [(int(start), int(end)) for start, end in offset_mapping]
    return None


def encode_focus_span(
    *,
    overlay: Any,
    text: str,
    focus_text: str,
    max_length: int,
    device: torch.device,
) -> SpanEncoding | None:
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    offset_mapping = None
    try:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = _tensor_offsets_to_list(encoded.pop("offset_mapping", None))
    except TypeError:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    batch = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    input_ids = [int(token) for token in batch["input_ids"][0].tolist()]
    span_match = match_anchor_span(
        text=text,
        anchor_text=focus_text,
        input_ids=input_ids,
        tokenizer=tokenizer,
        offsets=offset_mapping,
    )
    if span_match is None:
        return None
    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    matched_ids = input_ids[span_match.token_start : span_match.token_end + 1]
    return SpanEncoding(
        text=text,
        focus_text=focus_text,
        input_ids=input_ids,
        attention_mask=batch.get("attention_mask"),
        hidden_states=tuple(outputs.hidden_states),
        span_match=span_match,
        raw_tokens=decode_token_surfaces(tokenizer, matched_ids),
        decoded_tokens=decode_token_pieces(tokenizer, matched_ids),
    )


def span_mean_hidden(
    hidden_state: torch.Tensor,
    *,
    token_start: int,
    token_end: int,
) -> torch.Tensor:
    return hidden_state[token_start : token_end + 1].to(dtype=torch.float32).mean(dim=0)


def span_mean_hidden_for_layer(
    encoding: SpanEncoding,
    *,
    layer: int,
) -> torch.Tensor:
    hidden_state_index = int(layer) + 1
    return span_mean_hidden(
        encoding.hidden_states[hidden_state_index][0],
        token_start=int(encoding.span_match.token_start),
        token_end=int(encoding.span_match.token_end),
    )


def compute_neutral_basis(
    vectors: list[torch.Tensor],
    *,
    max_components: int = 3,
    variance_cutoff: float = 0.5,
) -> torch.Tensor | None:
    valid = [vector.to(dtype=torch.float32) for vector in vectors if vector is not None and vector.numel() > 0]
    if len(valid) < 2:
        return None
    matrix = torch.stack(valid, dim=0)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if int(centered.size(0)) < 2:
        return None
    try:
        _, singular_values, right = torch.pca_lowrank(centered, q=min(max_components, centered.size(0), centered.size(1)))
        energy = singular_values.square()
        if float(energy.sum().item()) <= _EPS:
            return None
        running = 0.0
        keep = 0
        total = float(energy.sum().item())
        for value in energy.tolist():
            running += float(value)
            keep += 1
            if running / total >= variance_cutoff:
                break
        keep = max(1, min(keep, right.size(1)))
        return right[:, :keep].to(dtype=torch.float32)
    except Exception:
        return None


def build_neutral_basis_by_layer(
    *,
    layers: list[int],
    case_names: list[str],
    encodings: dict[str, SpanEncoding],
    max_components: int = 3,
    variance_cutoff: float = 0.5,
) -> dict[int, torch.Tensor | None]:
    payload: dict[int, torch.Tensor | None] = {}
    for layer in layers:
        neutral_vectors = [
            span_mean_hidden_for_layer(encodings[case_name], layer=int(layer))
            for case_name in case_names
        ]
        payload[int(layer)] = compute_neutral_basis(
            neutral_vectors,
            max_components=max_components,
            variance_cutoff=variance_cutoff,
        )
    return payload


def project_out_basis(vector: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    projected = vector.to(dtype=torch.float32)
    if basis is None or basis.numel() == 0:
        return projected
    basis_local = basis.to(device=projected.device, dtype=torch.float32)
    for idx in range(basis_local.size(1)):
        direction = basis_local[:, idx]
        norm = float(direction.norm().item())
        if norm <= _EPS:
            continue
        unit = direction / norm
        projected = projected - torch.dot(projected, unit) * unit
    return projected


def cosine_or_none(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None:
        return None
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm <= _EPS or right_norm <= _EPS:
        return None
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item())


def build_group_concept_vectors(
    *,
    layers: list[int],
    name_to_group: dict[str, str],
    encodings: dict[str, SpanEncoding],
    neutral_basis_by_layer: dict[int, torch.Tensor | None],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, float | None]]]:
    layer_group_vectors: dict[int, dict[str, torch.Tensor]] = {}
    layer_group_norms: dict[int, dict[str, float | None]] = {}
    all_groups = sorted(set(name_to_group.values()))
    for layer in layers:
        case_vectors = {
            name: span_mean_hidden_for_layer(encoding, layer=int(layer))
            for name, encoding in encodings.items()
        }
        global_mean = torch.stack(list(case_vectors.values()), dim=0).mean(dim=0)
        group_vectors: dict[str, torch.Tensor] = {}
        group_norms: dict[str, float | None] = {}
        for group in all_groups:
            selected = [
                case_vectors[name]
                for name, current_group in name_to_group.items()
                if current_group == group and name in case_vectors
            ]
            if not selected:
                continue
            group_mean = torch.stack(selected, dim=0).mean(dim=0)
            centered = group_mean - global_mean
            projected = project_out_basis(centered, neutral_basis_by_layer.get(int(layer)))
            group_vectors[group] = projected.detach().cpu()
            group_norms[group] = float(projected.norm().item())
        layer_group_vectors[int(layer)] = group_vectors
        layer_group_norms[int(layer)] = group_norms
    return layer_group_vectors, layer_group_norms
