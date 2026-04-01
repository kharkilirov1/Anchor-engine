from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import QwenAnchorGeometryCase, make_qwen_anchor_geometry_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import (
    build_computability_flags,
    compute_cross_prompt_stability,
    compute_geometry_metrics,
    compute_mean_direction,
    decode_token_pieces,
    decode_token_surfaces,
    extract_delta_vectors,
    list_model_layers,
    match_anchor_span,
    token_has_leading_whitespace,
)


MODE_ORDER = ("full_span", "trimmed_span")
CLASS_ORDER = ("content_like", "procedure_like")
PRIMARY_METRICS = (
    "adjacent_cosine_coherence",
    "path_tortuosity",
    "rank1_explained_variance",
    "mean_direction_norm",
    "mean_step_norm",
    "curvature_proxy",
)
CURVE_METRICS = (
    "adjacent_cosine_coherence",
    "path_tortuosity",
    "rank1_explained_variance",
    "cross_prompt_stability",
)


def _to_scalar(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _fmt_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _summarize_numeric(values: list[float | None]) -> dict[str, float | int | None]:
    filtered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not filtered:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean_value = statistics.fmean(filtered)
    std_value = statistics.pstdev(filtered) if len(filtered) > 1 else 0.0
    return {
        "count": len(filtered),
        "mean": float(mean_value),
        "std": float(std_value),
        "min": float(min(filtered)),
        "max": float(max(filtered)),
    }


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


def encode_case(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]] | None]:
    if overlay.tokenizer is None:
        raise ValueError("tokenizer is required for geometry probe")
    offset_mapping = None
    try:
        encoded = overlay.tokenizer(
            case.prompt,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = _tensor_offsets_to_list(encoded.pop("offset_mapping", None))
    except TypeError:
        encoded = overlay.tokenizer(
            case.prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    batch = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    return batch, offset_mapping


def build_mode_span(
    *,
    mode: str,
    span_start: int,
    span_end: int,
    input_ids: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    start = span_start + 1 if mode == "trimmed_span" else span_start
    if start > span_end:
        return {
            "status": "unavailable",
            "reason": "trimmed_span_empty",
        }
    token_ids = input_ids[start : span_end + 1]
    raw_tokens = decode_token_surfaces(tokenizer, token_ids)
    decoded_tokens = decode_token_pieces(tokenizer, token_ids)
    leading_ws = None
    if token_ids:
        leading_ws = token_has_leading_whitespace(raw_tokens[0], decoded_tokens[0])
    return {
        "status": "ok",
        "token_start": int(start),
        "token_end": int(span_end),
        "token_count": int(len(token_ids)),
        "token_ids": [int(token_id) for token_id in token_ids],
        "raw_tokens": raw_tokens,
        "decoded_tokens": decoded_tokens,
        "decoded_span_text": tokenizer.decode(token_ids, skip_special_tokens=False),
        "first_token_has_leading_whitespace": leading_ws,
    }

def analyze_case_geometry(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    layers: list[int],
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    batch, offsets = encode_case(overlay=overlay, case=case, max_length=max_length, device=device)
    input_ids = [int(token) for token in batch["input_ids"][0].tolist()]
    span_match = match_anchor_span(
        text=case.prompt,
        anchor_text=case.anchor_text,
        input_ids=input_ids,
        tokenizer=overlay.tokenizer,
        offsets=offsets,
    )
    if span_match is None:
        return {
            "name": case.name,
            "anchor_class": case.anchor_class,
            "anchor_group": case.anchor_group,
            "anchor_text": case.anchor_text,
            "prompt": case.prompt,
            "description": case.description,
            "status": "skipped",
            "skip_reason": "anchor_span_not_matched_uniquely",
            "tokenization_audit": {
                "status": "skip",
                "issues": ["anchor_span_not_matched_uniquely"],
            },
            "modes": {},
        }

    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    matched_token_ids = input_ids[span_match.token_start : span_match.token_end + 1]
    matched_raw_tokens = decode_token_surfaces(overlay.tokenizer, matched_token_ids)
    matched_decoded_tokens = decode_token_pieces(overlay.tokenizer, matched_token_ids)
    modes: dict[str, Any] = {}
    for mode in MODE_ORDER:
        span_payload = build_mode_span(
            mode=mode,
            span_start=span_match.token_start,
            span_end=span_match.token_end,
            input_ids=input_ids,
            tokenizer=overlay.tokenizer,
        )
        if span_payload["status"] != "ok":
            modes[mode] = {
                "status": span_payload["status"],
                "reason": span_payload["reason"],
                "span": None,
                "layer_results": [],
            }
            continue
        layer_results: list[dict[str, Any]] = []
        for layer in layers:
            hidden_state_index = layer + 1
            layer_hidden = hidden_states[hidden_state_index][0]
            delta_vectors = extract_delta_vectors(
                hidden_states=layer_hidden,
                token_start=span_payload["token_start"],
                token_end=span_payload["token_end"],
            )
            metrics = compute_geometry_metrics(delta_vectors)
            mean_direction = compute_mean_direction(delta_vectors)
            layer_results.append(
                {
                    "layer": int(layer),
                    "hidden_state_index": int(hidden_state_index),
                    "metrics": {key: _to_scalar(value) for key, value in metrics.items()},
                    "computable_metrics": build_computability_flags(metrics),
                    "mean_direction": mean_direction.detach().cpu().tolist() if mean_direction is not None else None,
                }
            )
        modes[mode] = {
            "status": "ok",
            "span": span_payload,
            "layer_results": layer_results,
        }

    tokenization_audit = {
        "status": "clean",
        "issues": [],
        "notes": [],
        "anchor_text": case.anchor_text,
        "matched_text": span_match.matched_text,
        "match_method": span_match.match_method,
        "token_count": int(span_match.token_count),
        "token_ids": [int(token_id) for token_id in matched_token_ids],
        "raw_tokens": matched_raw_tokens,
        "decoded_tokens": matched_decoded_tokens,
        "first_token_has_leading_whitespace": token_has_leading_whitespace(
            matched_raw_tokens[0],
            matched_decoded_tokens[0],
        ) if matched_token_ids else None,
        "same_token_count_in_group": None,
        "same_token_ids_in_group": None,
        "same_decoded_tokens_in_group": None,
        "same_leading_whitespace_in_group": None,
        "trimmed_span_usable": modes["trimmed_span"]["status"] == "ok",
        "trimmed_span_has_full_geometry": (
            modes["trimmed_span"]["status"] == "ok"
            and int(modes["trimmed_span"]["span"]["token_count"]) >= 4
        ),
    }
    if int(span_match.token_count) < 4:
        tokenization_audit["notes"].append("full_span_short_for_path_metrics")
    if modes["trimmed_span"]["status"] != "ok":
        tokenization_audit["issues"].append("trimmed_span_unavailable")
        tokenization_audit["status"] = "noisy"
    elif int(modes["trimmed_span"]["span"]["token_count"]) < 4:
        tokenization_audit["notes"].append("trimmed_span_short_for_path_metrics")

    return {
        "name": case.name,
        "anchor_class": case.anchor_class,
        "anchor_group": case.anchor_group,
        "anchor_text": case.anchor_text,
        "prompt": case.prompt,
        "description": case.description,
        "status": "ok",
        "span_match": asdict(span_match),
        "tokenization_audit": tokenization_audit,
        "modes": modes,
    }


def finalize_tokenization_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_results = [result for result in results if result["status"] == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in valid_results:
        grouped[result["anchor_group"]].append(result)

    group_summary: dict[str, dict[str, Any]] = {}
    for group_name, group_results in grouped.items():
        audits = [result["tokenization_audit"] for result in group_results]
        token_counts = {int(audit["token_count"]) for audit in audits}
        token_id_sets = {tuple(int(token_id) for token_id in audit["token_ids"]) for audit in audits}
        decoded_token_sets = {tuple(audit["decoded_tokens"]) for audit in audits}
        leading_ws = {bool(audit["first_token_has_leading_whitespace"]) for audit in audits}
        same_token_count = len(token_counts) == 1
        same_token_ids = len(token_id_sets) == 1
        same_decoded_tokens = len(decoded_token_sets) == 1
        same_leading_whitespace = len(leading_ws) == 1
        group_status = "clean"
        if not all((same_token_count, same_token_ids, same_decoded_tokens, same_leading_whitespace)):
            group_status = "noisy"
        group_summary[group_name] = {
            "case_count": len(group_results),
            "status": group_status,
            "same_token_count": same_token_count,
            "same_token_ids": same_token_ids,
            "same_decoded_tokens": same_decoded_tokens,
            "same_leading_whitespace": same_leading_whitespace,
        }
        for result in group_results:
            audit = result["tokenization_audit"]
            audit["same_token_count_in_group"] = same_token_count
            audit["same_token_ids_in_group"] = same_token_ids
            audit["same_decoded_tokens_in_group"] = same_decoded_tokens
            audit["same_leading_whitespace_in_group"] = same_leading_whitespace
            if not same_token_count:
                audit["issues"].append("group_token_count_mismatch")
            if not same_token_ids:
                audit["issues"].append("group_token_id_mismatch")
            if not same_decoded_tokens:
                audit["issues"].append("group_decoded_token_mismatch")
            if not same_leading_whitespace:
                audit["issues"].append("group_leading_whitespace_mismatch")
            if audit["issues"]:
                audit["status"] = "noisy"

    skipped_results = [result for result in results if result["status"] != "ok"]
    summary = {
        "clean_case_count": sum(1 for result in valid_results if result["tokenization_audit"]["status"] == "clean"),
        "noisy_case_count": sum(1 for result in valid_results if result["tokenization_audit"]["status"] == "noisy"),
        "skip_case_count": len(skipped_results),
        "group_summary": group_summary,
        "groups_with_tokenization_mismatch": [
            group_name for group_name, info in group_summary.items() if info["status"] != "clean"
        ],
    }
    return summary


def _subset_results(
    results: list[dict[str, Any]],
    *,
    clean_only: bool,
) -> list[dict[str, Any]]:
    valid = [result for result in results if result["status"] == "ok"]
    if not clean_only:
        return valid
    return [result for result in valid if result["tokenization_audit"]["status"] == "clean"]


def _get_layer_result(
    result: dict[str, Any],
    *,
    mode: str,
    layer: int,
) -> dict[str, Any] | None:
    mode_payload = result["modes"].get(mode)
    if mode_payload is None or mode_payload["status"] != "ok":
        return None
    for layer_result in mode_payload["layer_results"]:
        if int(layer_result["layer"]) == int(layer):
            return layer_result
    return None


def _summarize_layer_metric(
    values: list[float | int | None],
) -> dict[str, float | int | None]:
    return _summarize_numeric([float(value) if value is not None else None for value in values])


def _group_cases_by_name(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["anchor_group"]].append(result)
    return grouped


def _direction_cosine(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> float | None:
    if left is None or right is None:
        return None
    left = left.to(dtype=torch.float32)
    right = right.to(dtype=torch.float32)
    if float(left.norm().item()) <= 1e-8 or float(right.norm().item()) <= 1e-8:
        return None
    return float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item())


def _collect_group_layer_vectors(
    group_results: list[dict[str, Any]],
    *,
    mode: str,
    layer: int,
) -> list[torch.Tensor]:
    directions: list[torch.Tensor] = []
    for result in group_results:
        layer_result = _get_layer_result(result, mode=mode, layer=layer)
        if layer_result is None:
            continue
        mean_direction = layer_result.get("mean_direction")
        if mean_direction is None:
            continue
        directions.append(torch.tensor(mean_direction, dtype=torch.float32))
    return directions


def _compute_group_aggregates(
    results: list[dict[str, Any]],
    *,
    mode: str,
    layers: list[int],
) -> dict[str, dict[str, Any]]:
    grouped = _group_cases_by_name(results)
    group_aggregates: dict[str, dict[str, Any]] = {}
    for group_name, group_results in sorted(grouped.items()):
        anchor_class = group_results[0]["anchor_class"]
        group_entry: dict[str, Any] = {
            "anchor_class": anchor_class,
            "case_names": [result["name"] for result in group_results],
            "case_count": len(group_results),
            "layers": [],
        }
        for layer in layers:
            metric_values: dict[str, list[float | int | None]] = defaultdict(list)
            directions = _collect_group_layer_vectors(group_results, mode=mode, layer=layer)
            available_cases = 0
            for result in group_results:
                layer_result = _get_layer_result(result, mode=mode, layer=layer)
                if layer_result is None:
                    continue
                available_cases += 1
                for metric_name, metric_value in layer_result["metrics"].items():
                    metric_values[metric_name].append(metric_value)
            stability = compute_cross_prompt_stability(directions)
            group_entry["layers"].append(
                {
                    "layer": int(layer),
                    "available_case_count": int(available_cases),
                    "metrics": {
                        metric_name: _summarize_layer_metric(values)
                        for metric_name, values in sorted(metric_values.items())
                    },
                    "cross_prompt_stability": {
                        key: _to_scalar(value)
                        for key, value in stability.items()
                    },
                    "group_mean_direction": (
                        torch.stack(directions, dim=0).mean(dim=0).tolist() if directions else None
                    ),
                }
            )
        group_aggregates[group_name] = group_entry
    return group_aggregates


def _compute_class_aggregates(
    results: list[dict[str, Any]],
    *,
    mode: str,
    layers: list[int],
    group_aggregates: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    class_aggregates: dict[str, dict[str, Any]] = {}
    for anchor_class in CLASS_ORDER:
        class_results = [result for result in results if result["anchor_class"] == anchor_class]
        class_groups = [
            (group_name, group_entry)
            for group_name, group_entry in group_aggregates.items()
            if group_entry["anchor_class"] == anchor_class
        ]
        layers_payload: list[dict[str, Any]] = []
        for layer in layers:
            metric_values: dict[str, list[float | int | None]] = defaultdict(list)
            class_directions: list[torch.Tensor] = []
            for result in class_results:
                layer_result = _get_layer_result(result, mode=mode, layer=layer)
                if layer_result is None:
                    continue
                for metric_name, metric_value in layer_result["metrics"].items():
                    metric_values[metric_name].append(metric_value)
                mean_direction = layer_result.get("mean_direction")
                if mean_direction is not None:
                    class_directions.append(torch.tensor(mean_direction, dtype=torch.float32))
            stability_values = []
            for _, group_entry in class_groups:
                layer_entry = next(
                    (item for item in group_entry["layers"] if int(item["layer"]) == int(layer)),
                    None,
                )
                if layer_entry is None:
                    continue
                stability_values.append(layer_entry["cross_prompt_stability"]["mean_pairwise_cosine"])
            layer_direction = torch.stack(class_directions, dim=0).mean(dim=0) if class_directions else None
            layers_payload.append(
                {
                    "layer": int(layer),
                    "case_count": len(class_results),
                    "metrics": {
                        metric_name: _summarize_layer_metric(values)
                        for metric_name, values in sorted(metric_values.items())
                    },
                    "cross_prompt_stability": _summarize_layer_metric(stability_values),
                    "class_mean_direction": layer_direction.tolist() if layer_direction is not None else None,
                }
            )
        class_aggregates[anchor_class] = {
            "case_count": len(class_results),
            "group_count": len(class_groups),
            "layers": layers_payload,
        }
    return class_aggregates


def _compute_group_polarity_margins(
    *,
    group_aggregates: dict[str, dict[str, Any]],
    class_aggregates: dict[str, dict[str, Any]],
    layers: list[int],
) -> None:
    class_centroids: dict[tuple[str, int], torch.Tensor | None] = {}
    for anchor_class, class_entry in class_aggregates.items():
        for layer_entry in class_entry["layers"]:
            mean_direction = layer_entry.get("class_mean_direction")
            class_centroids[(anchor_class, int(layer_entry["layer"]))] = (
                torch.tensor(mean_direction, dtype=torch.float32) if mean_direction is not None else None
            )
    for group_entry in group_aggregates.values():
        margins: list[float | None] = []
        for layer_entry in group_entry["layers"]:
            layer = int(layer_entry["layer"])
            group_direction_data = layer_entry.get("group_mean_direction")
            group_direction = (
                torch.tensor(group_direction_data, dtype=torch.float32)
                if group_direction_data is not None
                else None
            )
            content_centroid = class_centroids.get(("content_like", layer))
            procedure_centroid = class_centroids.get(("procedure_like", layer))
            content_affinity = _direction_cosine(group_direction, content_centroid)
            procedure_affinity = _direction_cosine(group_direction, procedure_centroid)
            polarity_margin = None
            if content_affinity is not None and procedure_affinity is not None:
                polarity_margin = float(content_affinity - procedure_affinity)
            layer_entry["content_affinity"] = _to_scalar(content_affinity)
            layer_entry["procedure_affinity"] = _to_scalar(procedure_affinity)
            layer_entry["polarity_margin"] = _to_scalar(polarity_margin)
            margins.append(polarity_margin)
        valid_margins = [(layer, margin) for layer, margin in zip(layers, margins) if margin is not None]
        sign_sequence = [
            1 if float(margin) > 1e-6 else -1 if float(margin) < -1e-6 else 0
            for _, margin in valid_margins
        ]
        sign_changes = sum(
            1
            for left, right in zip(sign_sequence, sign_sequence[1:])
            if left != 0 and right != 0 and left != right
        )
        strongest_content = max(valid_margins, key=lambda item: item[1], default=None)
        strongest_procedure = min(valid_margins, key=lambda item: item[1], default=None)
        group_entry["transition_summary"] = {
            "valid_layer_count": len(valid_margins),
            "sign_changes": int(sign_changes),
            "transitional": bool(sign_changes > 0),
            "first_content_like_layer": next((layer for layer, margin in valid_margins if margin > 0), None),
            "first_procedure_like_layer": next((layer for layer, margin in valid_margins if margin < 0), None),
            "strongest_content_like_layer": strongest_content[0] if strongest_content is not None else None,
            "strongest_content_like_margin": _to_scalar(strongest_content[1] if strongest_content is not None else None),
            "strongest_procedure_like_layer": strongest_procedure[0] if strongest_procedure is not None else None,
            "strongest_procedure_like_margin": _to_scalar(strongest_procedure[1] if strongest_procedure is not None else None),
        }


def _empty_layer_comparison(layer: int) -> dict[str, Any]:
    return {
        "layer": int(layer),
        "content_like": {},
        "procedure_like": {},
        "gaps": {},
        "directional_gaps": {},
        "positive_signals": 0,
        "available_signals": 0,
        "separation_score": None,
    }


def _layer_metric_mean(
    class_aggregates: dict[str, dict[str, Any]],
    *,
    anchor_class: str,
    layer: int,
    metric_name: str,
) -> float | None:
    class_entry = class_aggregates.get(anchor_class)
    if class_entry is None:
        return None
    layer_entry = next((item for item in class_entry["layers"] if int(item["layer"]) == int(layer)), None)
    if layer_entry is None:
        return None
    if metric_name == "cross_prompt_stability":
        return layer_entry["cross_prompt_stability"]["mean"]
    metric_summary = layer_entry["metrics"].get(metric_name)
    if metric_summary is None:
        return None
    return metric_summary["mean"]


def _compute_layer_comparisons(
    class_aggregates: dict[str, dict[str, Any]],
    *,
    layers: list[int],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    expected_orientations = {
        "adjacent_cosine_coherence": 1.0,
        "path_tortuosity": -1.0,
        "rank1_explained_variance": 1.0,
        "cross_prompt_stability": 1.0,
    }
    tracked_metrics = (
        "adjacent_cosine_coherence",
        "path_tortuosity",
        "rank1_explained_variance",
        "mean_direction_norm",
        "cross_prompt_stability",
    )
    for layer in layers:
        comparison = _empty_layer_comparison(layer)
        for metric_name in tracked_metrics:
            content_value = _layer_metric_mean(
                class_aggregates,
                anchor_class="content_like",
                layer=layer,
                metric_name=metric_name,
            )
            procedure_value = _layer_metric_mean(
                class_aggregates,
                anchor_class="procedure_like",
                layer=layer,
                metric_name=metric_name,
            )
            comparison["content_like"][metric_name] = _to_scalar(content_value)
            comparison["procedure_like"][metric_name] = _to_scalar(procedure_value)
            if content_value is None or procedure_value is None:
                comparison["gaps"][metric_name] = None
                comparison["directional_gaps"][metric_name] = None
                continue
            gap = float(content_value - procedure_value)
            comparison["gaps"][metric_name] = float(gap)
            orientation = expected_orientations.get(metric_name)
            if orientation is None:
                comparison["directional_gaps"][metric_name] = None
                continue
            directional_gap = float(gap * orientation)
            comparison["directional_gaps"][metric_name] = directional_gap
            comparison["available_signals"] += 1
            if directional_gap > 0:
                comparison["positive_signals"] += 1
        if comparison["available_signals"] > 0:
            comparison["separation_score"] = float(
                comparison["positive_signals"] / comparison["available_signals"]
            )
        comparisons.append(comparison)
    return comparisons


def _max_separation_layer(layer_comparisons: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in layer_comparisons if item["separation_score"] is not None]
    if not valid:
        return None
    return max(
        valid,
        key=lambda item: (
            float(item["separation_score"]),
            float(item["directional_gaps"].get("adjacent_cosine_coherence") or -999.0),
            float(item["directional_gaps"].get("cross_prompt_stability") or -999.0),
        ),
    )


def _first_positive_layer(layer_comparisons: list[dict[str, Any]]) -> int | None:
    for item in layer_comparisons:
        score = item["separation_score"]
        if score is not None and float(score) >= 0.75:
            return int(item["layer"])
    return None


def _stable_birth_layer(layer_comparisons: list[dict[str, Any]], run_length: int = 3) -> int | None:
    positives = [
        item
        for item in layer_comparisons
        if item["separation_score"] is not None and float(item["separation_score"]) >= 0.75
    ]
    if len(positives) < run_length:
        return None
    for idx in range(len(positives) - run_length + 1):
        window = positives[idx : idx + run_length]
        window_layers = [int(item["layer"]) for item in window]
        if window_layers == list(range(window_layers[0], window_layers[0] + run_length)):
            return window_layers[0]
    return None


def _compute_mode_verdict(layer_comparisons: list[dict[str, Any]]) -> str:
    valid = [item for item in layer_comparisons if item["separation_score"] is not None]
    if not valid:
        return "no_signal"
    strong_layers = [item for item in valid if float(item["separation_score"]) >= 0.75]
    weak_layers = [item for item in valid if float(item["separation_score"]) >= 0.50]
    strong_ratio = len(strong_layers) / max(len(valid), 1)
    weak_ratio = len(weak_layers) / max(len(valid), 1)
    if strong_ratio >= 0.50:
        return "clear_separation"
    if weak_ratio >= 0.25:
        return "partial_separation"
    return "no_separation"


def aggregate_mode_results(
    results: list[dict[str, Any]],
    *,
    mode: str,
    layers: list[int],
    clean_only: bool,
) -> dict[str, Any]:
    subset = _subset_results(results, clean_only=clean_only)
    group_aggregates = _compute_group_aggregates(subset, mode=mode, layers=layers)
    class_aggregates = _compute_class_aggregates(
        subset,
        mode=mode,
        layers=layers,
        group_aggregates=group_aggregates,
    )
    _compute_group_polarity_margins(
        group_aggregates=group_aggregates,
        class_aggregates=class_aggregates,
        layers=layers,
    )
    layer_comparisons = _compute_layer_comparisons(class_aggregates, layers=layers)
    max_layer = _max_separation_layer(layer_comparisons)
    positive_layer = _first_positive_layer(layer_comparisons)
    stable_birth = _stable_birth_layer(layer_comparisons)
    return {
        "subset": "clean_only" if clean_only else "all_valid",
        "case_count": len(subset),
        "group_count": len(group_aggregates),
        "class_aggregates": class_aggregates,
        "group_aggregates": group_aggregates,
        "layer_comparisons": layer_comparisons,
        "max_separation_layer": (
            {
                "layer": int(max_layer["layer"]),
                "separation_score": _to_scalar(max_layer["separation_score"]),
                "positive_signals": int(max_layer["positive_signals"]),
                "available_signals": int(max_layer["available_signals"]),
            }
            if max_layer is not None
            else None
        ),
        "first_positive_layer": positive_layer,
        "stable_birth_layer": stable_birth,
        "verdict": _compute_mode_verdict(layer_comparisons),
    }


def aggregate_results(results: list[dict[str, Any]], *, layers: list[int]) -> dict[str, Any]:
    tokenization_summary = finalize_tokenization_audit(results)
    return {
        "tokenization_summary": tokenization_summary,
        "modes": {
            mode: {
                "all_valid": aggregate_mode_results(results, mode=mode, layers=layers, clean_only=False),
                "clean_only": aggregate_mode_results(results, mode=mode, layers=layers, clean_only=True),
            }
            for mode in MODE_ORDER
        },
    }


def infer_overall_interpretation(aggregate: dict[str, Any]) -> dict[str, Any]:
    full_clean = aggregate["modes"]["full_span"]["clean_only"]
    trimmed_clean = aggregate["modes"]["trimmed_span"]["clean_only"]
    full_verdict = full_clean["verdict"]
    trimmed_verdict = trimmed_clean["verdict"]
    if full_verdict == "clear_separation" and trimmed_verdict == "clear_separation":
        support = "supported"
    elif full_verdict in {"clear_separation", "partial_separation"} and trimmed_verdict in {
        "clear_separation",
        "partial_separation",
    }:
        support = "partially_supported"
    else:
        support = "not_supported"
    noisy_case_count = int(aggregate["tokenization_summary"]["noisy_case_count"])
    clean_case_count = int(aggregate["tokenization_summary"]["clean_case_count"])
    return {
        "full_span_clean_verdict": full_verdict,
        "trimmed_span_clean_verdict": trimmed_verdict,
        "support_after_tokenization_controls": support,
        "tokenization_effect_substantial": bool(noisy_case_count > clean_case_count),
    }


def strip_mean_directions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for result in results:
        cloned = json.loads(json.dumps(result))
        if cloned.get("status") == "ok":
            for mode in MODE_ORDER:
                mode_payload = cloned["modes"].get(mode)
                if not mode_payload or mode_payload["status"] != "ok":
                    continue
                for layer_result in mode_payload["layer_results"]:
                    layer_result.pop("mean_direction", None)
        stripped.append(cloned)
    return stripped


def _layer_header() -> str:
    return (
        "| layer | content coherence | procedure coherence | gap | "
        "content tortuosity | procedure tortuosity | gap | "
        "content rank1 EV | procedure rank1 EV | gap | "
        "content stability | procedure stability | gap |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )


def _class_curve_rows(mode_result: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in mode_result["layer_comparisons"]:
        layer = int(item["layer"])
        rows.append(
            "| "
            + " | ".join(
                [
                    f"L{layer:02d}",
                    _fmt_metric(item["content_like"].get("adjacent_cosine_coherence")),
                    _fmt_metric(item["procedure_like"].get("adjacent_cosine_coherence")),
                    _fmt_metric(item["gaps"].get("adjacent_cosine_coherence")),
                    _fmt_metric(item["content_like"].get("path_tortuosity")),
                    _fmt_metric(item["procedure_like"].get("path_tortuosity")),
                    _fmt_metric(item["gaps"].get("path_tortuosity")),
                    _fmt_metric(item["content_like"].get("rank1_explained_variance")),
                    _fmt_metric(item["procedure_like"].get("rank1_explained_variance")),
                    _fmt_metric(item["gaps"].get("rank1_explained_variance")),
                    _fmt_metric(item["content_like"].get("cross_prompt_stability")),
                    _fmt_metric(item["procedure_like"].get("cross_prompt_stability")),
                    _fmt_metric(item["gaps"].get("cross_prompt_stability")),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _group_curve_rows(mode_result: dict[str, Any]) -> str:
    header = (
        "| group | class | transitional | strongest content layer | strongest procedure layer | margin curve |\n"
        "| --- | --- | --- | ---: | ---: | --- |"
    )
    rows = [header]
    for group_name, group_entry in sorted(mode_result["group_aggregates"].items()):
        transition = group_entry["transition_summary"]
        curve = " ".join(
            f"L{int(layer_entry['layer']):02d}:{_fmt_metric(layer_entry.get('polarity_margin'))}"
            for layer_entry in group_entry["layers"]
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    group_entry["anchor_class"],
                    "yes" if transition["transitional"] else "no",
                    str(transition["strongest_content_like_layer"] if transition["strongest_content_like_layer"] is not None else "n/a"),
                    str(transition["strongest_procedure_like_layer"] if transition["strongest_procedure_like_layer"] is not None else "n/a"),
                    curve,
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _tokenization_audit_rows(results: list[dict[str, Any]]) -> str:
    header = (
        "| case | class | group | audit | match | token_count | leading_ws | same_count | same_ids | trimmed_full_geometry | decoded_tokens | issues |\n"
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |"
    )
    rows = [header]
    for result in results:
        audit = result["tokenization_audit"]
        rows.append(
            "| "
            + " | ".join(
                [
                    result["name"],
                    result["anchor_class"],
                    result["anchor_group"],
                    audit.get("status", "skip"),
                    audit.get("match_method", "n/a"),
                    str(audit.get("token_count", "n/a")),
                    str(audit.get("first_token_has_leading_whitespace", "n/a")),
                    str(audit.get("same_token_count_in_group", "n/a")),
                    str(audit.get("same_token_ids_in_group", "n/a")),
                    str(audit.get("trimmed_span_has_full_geometry", "n/a")),
                    "`" + " | ".join(audit.get("decoded_tokens", [])) + "`" if audit.get("decoded_tokens") else "n/a",
                    ", ".join(audit.get("issues", [])) if audit.get("issues") else "-",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def build_markdown_report(
    *,
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    interpretation: dict[str, Any],
    layers: list[int],
) -> str:
    tokenization = aggregate["tokenization_summary"]
    full_clean = aggregate["modes"]["full_span"]["clean_only"]
    trimmed_clean = aggregate["modes"]["trimmed_span"]["clean_only"]
    full_all = aggregate["modes"]["full_span"]["all_valid"]
    trimmed_all = aggregate["modes"]["trimmed_span"]["all_valid"]
    trimmed_full_geometry_count = sum(
        1
        for result in results
        if result["status"] == "ok"
        and bool(result["tokenization_audit"].get("trimmed_span_has_full_geometry"))
    )
    lines = [
        "# Qwen Anchor Geometry Report",
        "",
        "## Summary",
        "",
        "- Model: `Qwen/Qwen2.5-1.5B`",
        f"- Layer indices analyzed: `{layers[0]}..{layers[-1]}` ({len(layers)} total model layers; embedding state kept only for reference)",
        f"- Clean cases: `{tokenization['clean_case_count']}`",
        f"- Noisy cases: `{tokenization['noisy_case_count']}`",
        f"- Skipped cases: `{tokenization['skip_case_count']}`",
        f"- Full-span clean verdict: `{full_clean['verdict']}`",
        f"- Trimmed-span clean verdict: `{trimmed_clean['verdict']}`",
        f"- Support after tokenization controls: `{interpretation['support_after_tokenization_controls']}`",
        f"- Cases retaining full geometry after trimming: `{trimmed_full_geometry_count}` / `{len(results)}`",
        "",
        "## Tokenization audit table",
        "",
        _tokenization_audit_rows(results),
        "",
        "## Class-level layer curves — full span (clean only)",
        "",
        _layer_header(),
        _class_curve_rows(full_clean),
        "",
        "## Class-level layer curves — trimmed span (clean only)",
        "",
        _layer_header(),
        _class_curve_rows(trimmed_clean),
        "",
        "## Group-level layer curves — full span (clean only)",
        "",
        _group_curve_rows(full_clean),
        "",
        "## Group-level layer curves — trimmed span (clean only)",
        "",
        _group_curve_rows(trimmed_clean),
        "",
        "## Full-span vs trimmed-span comparison",
        "",
        "| subset | verdict | case_count | max separation layer | first positive layer | stable birth layer |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| full_span / all_valid | {full_all['verdict']} | {full_all['case_count']} | {full_all['max_separation_layer']['layer'] if full_all['max_separation_layer'] else 'n/a'} | {full_all['first_positive_layer'] if full_all['first_positive_layer'] is not None else 'n/a'} | {full_all['stable_birth_layer'] if full_all['stable_birth_layer'] is not None else 'n/a'} |",
        f"| full_span / clean_only | {full_clean['verdict']} | {full_clean['case_count']} | {full_clean['max_separation_layer']['layer'] if full_clean['max_separation_layer'] else 'n/a'} | {full_clean['first_positive_layer'] if full_clean['first_positive_layer'] is not None else 'n/a'} | {full_clean['stable_birth_layer'] if full_clean['stable_birth_layer'] is not None else 'n/a'} |",
        f"| trimmed_span / all_valid | {trimmed_all['verdict']} | {trimmed_all['case_count']} | {trimmed_all['max_separation_layer']['layer'] if trimmed_all['max_separation_layer'] else 'n/a'} | {trimmed_all['first_positive_layer'] if trimmed_all['first_positive_layer'] is not None else 'n/a'} | {trimmed_all['stable_birth_layer'] if trimmed_all['stable_birth_layer'] is not None else 'n/a'} |",
        f"| trimmed_span / clean_only | {trimmed_clean['verdict']} | {trimmed_clean['case_count']} | {trimmed_clean['max_separation_layer']['layer'] if trimmed_clean['max_separation_layer'] else 'n/a'} | {trimmed_clean['first_positive_layer'] if trimmed_clean['first_positive_layer'] is not None else 'n/a'} | {trimmed_clean['stable_birth_layer'] if trimmed_clean['stable_birth_layer'] is not None else 'n/a'} |",
        "",
        "## Layer of maximal separation",
        "",
        f"- Full span / clean only: `{full_clean['max_separation_layer']}`",
        f"- Trimmed span / clean only: `{trimmed_clean['max_separation_layer']}`",
        "",
        "## Whether evidence supports polarity-from-geometry",
        "",
        f"Current judgment: `{interpretation['support_after_tokenization_controls']}`.",
        "",
        "The strongest evidence should come from clean-only trimmed spans, because that setting removes the first-token boundary artifact while keeping the original tokenizer intact. If full-span and trimmed-span disagree, trimmed-span is treated as the stricter control.",
        "",
        "## Limitations",
        "",
        "- The prompt set is still small and local, so group-level conclusions can be noisy.",
        "- Trimming removes one token from every anchor; in this prompt set only the 5-token FastAPI group keeps enough tokens for coherence, tortuosity, and rank-1 EV after trimming.",
        "- Cross-prompt stability is defined within the paired prompts already present in the probe, not over a large paraphrase set.",
        "- Mean-direction affinity is a diagnostic proxy and not a causal proof of polarity.",
        "- Some groups may remain transitional because the phrase itself mixes content and procedure semantics.",
        "",
        "## Recommended next step",
        "",
        "Run a tightly paired paraphrase probe per anchor group, holding token count fixed while varying only the surrounding sentence frame.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    *,
    payload: dict[str, Any],
    report_text: str,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen anchor geometry probe.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--json-path",
        default=str(ROOT / "archive" / "qwen_anchor_geometry_probe.json"),
    )
    parser.add_argument(
        "--markdown-path",
        default=str(ROOT / "docs" / "research" / "qwen_anchor_geometry_report.md"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model_name,
        cfg=TOY_CONFIG,
        device=device,
    )
    overlay.eval()

    num_hidden_layers = int(getattr(overlay.base_model.config, "num_hidden_layers"))
    layers = list_model_layers(num_hidden_layers)
    cases = make_qwen_anchor_geometry_cases()
    raw_results = [
        analyze_case_geometry(
            overlay=overlay,
            case=case,
            layers=layers,
            max_length=int(args.max_length),
            device=device,
        )
        for case in cases
    ]
    aggregate = aggregate_results(raw_results, layers=layers)
    interpretation = infer_overall_interpretation(aggregate)
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": args.model_name,
            "device": str(device),
            "max_length": int(args.max_length),
        },
        "layers": layers,
        "hidden_state_mapping": {
            "embedding_state_index": 0,
            "model_layer_to_hidden_state_index": {str(layer): int(layer + 1) for layer in layers},
        },
        "cases": [asdict(case) for case in cases],
        "results": strip_mean_directions(raw_results),
        "aggregate": aggregate,
        "interpretation": interpretation,
    }
    report_text = build_markdown_report(
        results=raw_results,
        aggregate=aggregate,
        interpretation=interpretation,
        layers=layers,
    )
    write_outputs(
        payload=payload,
        report_text=report_text,
        json_path=Path(args.json_path),
        markdown_path=Path(args.markdown_path),
    )
    print(f"saved json: {args.json_path}")
    print(f"saved md: {args.markdown_path}")


if __name__ == "__main__":
    main()
