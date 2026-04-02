from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import (
    build_tail_reference_layers,
    compute_geometry_metrics,
    extract_delta_vectors,
    select_tail_probe_layers,
    match_anchor_span,
    decode_token_pieces,
    decode_token_surfaces,
    token_has_leading_whitespace,
)
from src.data.qwen_anchor_geometry_cases import (
    make_qwen_anchor_geometry_cases,
    QwenAnchorGeometryCase,
)


DEFAULT_PROBE_LAYER_COUNT = 10
DEFAULT_CONFLICT_THRESHOLD = 0.55
DEFAULT_BIAS_SCALE = 1.50
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_FREQUENCY_PENALTY = 0.05
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_MAX_BIAS_GATE_SUM = 1.50
DEFAULT_ENTROPY_TOP_K = 32
DEFAULT_ENTROPY_THRESHOLD = 0.35
DEFAULT_ENTROPY_SLOPE = 0.08
DEFAULT_PRESSURE_THRESHOLD = 0.60
DEFAULT_PRESSURE_SLOPE = 0.08
DEFAULT_PRESSURE_RESCUE_FLOOR = 0.20


def resolve_geometry_probe_spec(
    overlay: QwenAnchorOverlay,
) -> tuple[list[int], dict[str, int]]:
    num_hidden_layers = int(getattr(overlay, "model_num_hidden_layers", 0))
    if num_hidden_layers <= 0:
        raise ValueError("model must expose a positive num_hidden_layers value")
    probe_layers = select_tail_probe_layers(
        num_hidden_layers=num_hidden_layers,
        count=DEFAULT_PROBE_LAYER_COUNT,
    )
    reference_layers = build_tail_reference_layers(probe_layers)
    return probe_layers, reference_layers

KEYWORD_MAP: dict[str, dict[str, Any]] = {
    "strictly_vegan_meal_plan_policy": {
        "positive": ["vegan", "plant-based", "tofu", "lentil", "chickpea", "beans", "vegetable", "mushroom"],
        "negative": ["egg", "eggs", "cheese", "butter", "milk", "cream", "meat", "chicken", "beef"],
        "min_unique_positive_hits": 2,
    },
    "async_fastapi_service_architecture_policy": {
        "positive": ["async", "await", "FastAPI", "router", "endpoint", "dependency", "asyncio"],
        "negative": ["Flask", "Django", "jinja", "template rendering", "class-based view", "wsgi", "sync view", "sync handler"],
        "min_unique_positive_hits": 2,
    },
    "json_only_response_format_policy": {
        "positive": ["json", "JSON", "{", "}", "key", "value", "format"],
        "negative": ["markdown", "plain text", "prose", "sorry", "I cannot", "Here is"],
        "min_unique_positive_hits": 2,
    },
    "proof_by_contradiction_reasoning_steps": {
        "positive": ["assume", "contradiction", "suppose", "therefore", "absurd", "QED", "proof"],
        "negative": ["example", "for instance", "because", "simply", "just", "obviously"],
        "min_unique_positive_hits": 2,
    },
    "binary_search_update_loop_procedure": {
        "positive": ["mid", "low", "high", "left", "right", "while", "binary", "O(log"],
        "negative": ["for i", "linear", "scan", "iterate", "brute"],
        "min_unique_positive_hits": 2,
    },
    "dependency_injection_request_flow_sequence": {
        "positive": ["inject", "dependency", "container", "resolve", "provider", "interface"],
        "negative": ["global", "singleton", "import", "hardcode", "direct instantiation"],
        "min_unique_positive_hits": 2,
    },
}

DEGENERATE_BIGRAM_RATIO = 0.30
DEGENERATE_MAX_SENTENCE_REPEAT = 3


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


def _to_scalar(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _find_occurrences(text: str, term: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + len(term)
    return positions


def _count_negative_hits(
    text: str,
    term: str,
    protected_phrases: list[str],
) -> tuple[int, int]:
    occurrences = _find_occurrences(text, term)
    if not occurrences:
        return 0, 0
    protected_spans: list[tuple[int, int]] = []
    for phrase in protected_phrases:
        for start in _find_occurrences(text, phrase):
            protected_spans.append((start, start + len(phrase)))
    effective = 0
    protected = 0
    for start in occurrences:
        end = start + len(term)
        if any(span_start <= start and end <= span_end for span_start, span_end in protected_spans):
            protected += 1
        else:
            effective += 1
    return effective, protected


def compute_constraint_analysis(
    text: str,
    keyword_spec: dict[str, Any],
) -> dict[str, Any]:
    positive_keywords = [str(token).lower() for token in keyword_spec.get("positive", [])]
    negative_keywords = [str(token).lower() for token in keyword_spec.get("negative", [])]
    negative_exceptions = {
        str(token).lower(): [str(phrase).lower() for phrase in phrases]
        for token, phrases in dict(keyword_spec.get("negative_exceptions", {})).items()
    }
    raw_metrics = analyze_keywords(text, positive_keywords=[], negative_keywords=[])
    lowered = text.lower()

    positive_hits = {token: lowered.count(token) for token in positive_keywords if lowered.count(token) > 0}
    negative_hits: dict[str, int] = {}
    protected_negative_hits: dict[str, int] = {}
    for token in negative_keywords:
        effective, protected = _count_negative_hits(
            lowered,
            token,
            protected_phrases=negative_exceptions.get(token, []),
        )
        if effective > 0:
            negative_hits[token] = effective
        if protected > 0:
            protected_negative_hits[token] = protected

    unique_positive_hits = len(positive_hits)
    negative_total = int(sum(negative_hits.values()))
    degenerate_output = bool(
        float(raw_metrics["repeated_bigram_ratio"]) >= DEGENERATE_BIGRAM_RATIO
        or int(raw_metrics["max_sentence_repeat"]) >= DEGENERATE_MAX_SENTENCE_REPEAT
    )
    min_unique_positive_hits = int(keyword_spec.get("min_unique_positive_hits", 2))
    constraint_satisfied = bool(
        unique_positive_hits >= min_unique_positive_hits
        and negative_total == 0
        and not degenerate_output
    )
    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "protected_negative_hits": protected_negative_hits,
        "positive_total": int(sum(positive_hits.values())),
        "negative_total": negative_total,
        "unique_positive_hits": unique_positive_hits,
        "min_unique_positive_hits": min_unique_positive_hits,
        "lexical_score": float(raw_metrics["lexical_score"]),
        "repeated_bigram_ratio": float(raw_metrics["repeated_bigram_ratio"]),
        "max_sentence_repeat": int(raw_metrics["max_sentence_repeat"]),
        "degeneracy_penalty": float(raw_metrics["degeneracy_penalty"]),
        "quality_score": float(raw_metrics["quality_score"]),
        "degenerate_output": degenerate_output,
        "constraint_satisfied": constraint_satisfied,
        "constraint_score": float(1.0 if constraint_satisfied else 0.0),
        "drift_detected": bool(negative_total > 0),
    }


def encode_case(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]] | None]:
    if overlay.tokenizer is None:
        raise ValueError("tokenizer is required")
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


def generate_base(
    overlay: QwenAnchorOverlay,
    prompt: str,
    max_new_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    device = next(overlay.parameters()).device
    encoded = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generated = input_ids
    generated_mask = attention_mask
    steps: list[dict[str, Any]] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = overlay.base_model(
                input_ids=generated,
                attention_mask=generated_mask,
                return_dict=True,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            generated_mask = torch.cat(
                [generated_mask, torch.ones((1, 1), device=device, dtype=generated_mask.dtype)],
                dim=1,
            )
            token_id = int(next_token.item())
            steps.append(
                {
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                }
            )
            if token_id == int(getattr(tokenizer, "eos_token_id", -1)):
                break
            if generated.size(1) >= max_length:
                break
    continuation_ids = generated[0, input_ids.size(1):]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "continuation_text": continuation_text,
        "steps": steps,
    }


def analyze_keywords(
    text: str,
    positive_keywords: list[str],
    negative_keywords: list[str],
) -> dict[str, Any]:
    lowered = text.lower()
    positive_hits = {token: lowered.count(token.lower()) for token in positive_keywords if token.lower() in lowered}

    protected_negative_hits: dict[str, int] = {}
    negative_hits: dict[str, int] = {}
    for token in negative_keywords:
        lowered_token = token.lower()
        if lowered_token not in lowered:
            continue
        total = lowered.count(lowered_token)
        protected = 0
        start = 0
        while True:
            idx = lowered.find(lowered_token, start)
            if idx < 0:
                break
            prefix = lowered[max(0, idx - 12):idx]
            if "vegan " in prefix or "plant-based " in prefix:
                protected += 1
            start = idx + len(lowered_token)
        effective = total - protected
        if effective > 0:
            negative_hits[token] = effective
        if protected > 0:
            protected_negative_hits[token] = protected

    lexical_score = float(sum(positive_hits.values()) - sum(negative_hits.values()))
    word_tokens = re.findall(r"[a-zA-Z_]+", lowered)
    bigrams = list(zip(word_tokens, word_tokens[1:]))
    repeated_bigram_ratio = 0.0
    if bigrams:
        repeated_bigram_ratio = 1.0 - (len(set(bigrams)) / max(len(bigrams), 1))
    sentence_candidates = [
        sentence.strip()
        for sentence in re.split(r"[.!?\n]+", lowered)
        if sentence.strip()
    ]
    sentence_counts: dict[str, int] = {}
    max_sentence_repeat = 0
    for sentence in sentence_candidates:
        sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
        max_sentence_repeat = max(max_sentence_repeat, sentence_counts[sentence])
    degeneracy_penalty = float(8.0 * repeated_bigram_ratio + 1.5 * max(0, max_sentence_repeat - 1))
    quality_score = float(lexical_score - degeneracy_penalty - 1.5 * sum(negative_hits.values()))
    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "protected_negative_hits": protected_negative_hits,
        "positive_total": int(sum(positive_hits.values())),
        "negative_total": int(sum(negative_hits.values())),
        "lexical_score": lexical_score,
        "repeated_bigram_ratio": float(repeated_bigram_ratio),
        "max_sentence_repeat": int(max_sentence_repeat),
        "degeneracy_penalty": float(degeneracy_penalty),
        "quality_score": float(quality_score),
    }


def _extract_continuation_text(payload: dict[str, Any]) -> str:
    continuation_text = payload.get("continuation_text")
    if isinstance(continuation_text, str):
        return continuation_text
    generated_text = payload.get("generated_text")
    prompt = payload.get("prompt")
    if isinstance(generated_text, str) and isinstance(prompt, str) and generated_text.startswith(prompt):
        return generated_text[len(prompt):].lstrip()
    return generated_text if isinstance(generated_text, str) else ""


def compute_geometry_profile(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    max_length: int,
    device: torch.device,
    probe_layers: list[int],
    reference_layers: dict[str, int],
) -> dict[str, Any] | None:
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
        return None

    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    matched_token_ids = input_ids[span_match.token_start: span_match.token_end + 1]
    raw_tokens = decode_token_surfaces(overlay.tokenizer, matched_token_ids)
    decoded_tokens = decode_token_pieces(overlay.tokenizer, matched_token_ids)
    rank1_profile: dict[str, float | None] = {}
    tortuosity_profile: dict[str, float | None] = {}
    layer_metrics: dict[str, dict[str, float | int | None]] = {}
    for layer in probe_layers:
        delta_vectors = extract_delta_vectors(
            hidden_states[layer + 1][0],
            span_match.token_start,
            span_match.token_end,
        )
        metrics = compute_geometry_metrics(delta_vectors)
        rank1_profile[str(layer)] = _to_scalar(metrics.get("rank1_explained_variance"))
        tortuosity_profile[str(layer)] = _to_scalar(metrics.get("path_tortuosity"))
        layer_metrics[str(layer)] = {key: _to_scalar(value) for key, value in metrics.items()}

    mature_layer = int(reference_layers["mature_layer"])
    template_prev_layer = int(reference_layers["template_prev_layer"])
    template_curr_layer = int(reference_layers["template_curr_layer"])
    slope_start_layer = int(reference_layers["slope_start_layer"])
    slope_end_layer = int(reference_layers["slope_end_layer"])

    r1_reference = rank1_profile.get(str(mature_layer))
    r1_template_prev = rank1_profile.get(str(template_prev_layer))
    r1_template_curr = rank1_profile.get(str(template_curr_layer))
    delta_template_pair = None
    if r1_template_prev is not None and r1_template_curr is not None:
        delta_template_pair = float(r1_template_curr - r1_template_prev)

    slope_tail_window = None
    slope_points = [
        (layer, rank1_profile[str(layer)])
        for layer in range(slope_start_layer, slope_end_layer + 1)
        if rank1_profile.get(str(layer)) is not None
    ]
    if len(slope_points) >= 2:
        xs = np.array([point[0] for point in slope_points], dtype=np.float64)
        ys = np.array([float(point[1]) for point in slope_points], dtype=np.float64)
        slope_tail_window = float(np.polyfit(xs, ys, deg=1)[0])

    if r1_reference is not None and float(r1_reference) > 0.65:
        anchor_cluster = "mature"
    elif delta_template_pair is not None and float(delta_template_pair) > 0.08:
        anchor_cluster = "template"
    else:
        anchor_cluster = "flat"

    return {
        "span_match": asdict(span_match),
        "matched_token_ids": [int(token_id) for token_id in matched_token_ids],
        "raw_tokens": raw_tokens,
        "decoded_tokens": decoded_tokens,
        "first_token_has_leading_whitespace": token_has_leading_whitespace(raw_tokens[0], decoded_tokens[0]) if raw_tokens else None,
        "rank1_profile": rank1_profile,
        "tortuosity_profile": tortuosity_profile,
        "layer_metrics": layer_metrics,
        "probe_layers": [int(layer) for layer in probe_layers],
        "reference_layers": {key: int(value) for key, value in reference_layers.items()},
        "r1_reference": _to_scalar(r1_reference),
        "delta_template_pair": _to_scalar(delta_template_pair),
        "slope_tail_window": _to_scalar(slope_tail_window),
        "anchor_cluster": anchor_cluster,
    }


def analyze_case(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    *,
    probe_layers: list[int],
    reference_layers: dict[str, int],
    max_length: int,
    max_new_tokens: int,
    conflict_threshold: float,
    bias_scale: float,
    repetition_penalty: float,
    frequency_penalty: float,
    no_repeat_ngram_size: int,
    max_bias_gate_sum: float,
    entropy_top_k: int,
    entropy_threshold: float,
    entropy_slope: float,
    pressure_threshold: float,
    pressure_slope: float,
    pressure_rescue_floor: float,
    device: torch.device,
) -> dict[str, Any] | None:
    geometry = compute_geometry_profile(
        overlay=overlay,
        case=case,
        max_length=max_length,
        device=device,
        probe_layers=probe_layers,
        reference_layers=reference_layers,
    )
    if geometry is None:
        print(f"SKIP {case.name}: span not matched")
        return None

    keyword_spec = KEYWORD_MAP.get(case.anchor_group)
    base = generate_base(
        overlay=overlay,
        prompt=case.prompt,
        max_new_tokens=max_new_tokens,
        max_length=max_length * 2,
    )
    with torch.inference_mode():
        anchor = overlay.generate_with_anchor_bias(
            prompt=case.prompt,
            max_new_tokens=max_new_tokens,
            max_length=max_length * 2,
            conflict_threshold=conflict_threshold,
            bias_scale=bias_scale,
            greedy=True,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_bias_gate_sum=max_bias_gate_sum,
            entropy_top_k=entropy_top_k,
            entropy_threshold=entropy_threshold,
            entropy_slope=entropy_slope,
            pressure_threshold=pressure_threshold,
            pressure_slope=pressure_slope,
            pressure_rescue_floor=pressure_rescue_floor,
        )

    base_continuation = _extract_continuation_text(base)
    anchor_continuation = _extract_continuation_text(anchor)

    if keyword_spec is None:
        base_analysis = {
            "positive_total": 0,
            "negative_total": 0,
            "unique_positive_hits": 0,
            "quality_score": None,
            "constraint_score": None,
            "constraint_satisfied": False,
            "degenerate_output": False,
            "drift_detected": False,
        }
        anchor_analysis = {
            "positive_total": 0,
            "negative_total": 0,
            "unique_positive_hits": 0,
            "quality_score": None,
            "constraint_score": None,
            "constraint_satisfied": False,
            "degenerate_output": False,
            "drift_detected": False,
        }
        constraint_delta = None
    else:
        base_analysis = compute_constraint_analysis(base_continuation, keyword_spec=keyword_spec)
        anchor_analysis = compute_constraint_analysis(anchor_continuation, keyword_spec=keyword_spec)
        constraint_delta = float(anchor_analysis["constraint_score"] - base_analysis["constraint_score"])

    print(f"DONE {case.name}: cluster={geometry['anchor_cluster']}")
    return {
        "name": case.name,
        "anchor_class": case.anchor_class,
        "anchor_group": case.anchor_group,
        "anchor_text": case.anchor_text,
        "span_match": geometry["span_match"],
        "matched_token_ids": geometry["matched_token_ids"],
        "raw_tokens": geometry["raw_tokens"],
        "decoded_tokens": geometry["decoded_tokens"],
        "first_token_has_leading_whitespace": geometry["first_token_has_leading_whitespace"],
        "rank1_profile": geometry["rank1_profile"],
        "tortuosity_profile": geometry["tortuosity_profile"],
        "probe_layers": geometry["probe_layers"],
        "reference_layers": geometry["reference_layers"],
        "r1_reference": geometry["r1_reference"],
        "delta_template_pair": geometry["delta_template_pair"],
        "slope_tail_window": geometry["slope_tail_window"],
        "anchor_cluster": geometry["anchor_cluster"],
        "base_continuation": base_continuation,
        "anchor_continuation": anchor_continuation,
        "base_analysis": base_analysis,
        "anchor_analysis": anchor_analysis,
        "base_degenerate": bool(base_analysis["degenerate_output"]),
        "included_in_calibration": bool(not base_analysis["degenerate_output"] and constraint_delta is not None),
        "constraint_delta": _to_scalar(constraint_delta),
    }


def _cluster_summary(
    cases: list[dict[str, Any]],
    cluster: str,
    *,
    selector: str = "included_in_calibration",
) -> dict[str, Any]:
    cluster_cases = [case for case in cases if case["anchor_cluster"] == cluster]
    if selector == "all":
        selected_cases = list(cluster_cases)
    elif selector == "base_degenerate":
        selected_cases = [case for case in cluster_cases if case.get("base_degenerate", False)]
    else:
        selected_cases = [case for case in cluster_cases if case.get("included_in_calibration", False)]
    deltas = [float(case["constraint_delta"]) for case in selected_cases if case["constraint_delta"] is not None]
    drifts = [1.0 if case["anchor_analysis"]["drift_detected"] else 0.0 for case in selected_cases]
    r1_values = [float(case["r1_reference"]) for case in cluster_cases if case["r1_reference"] is not None]
    excluded_case_names = [case["name"] for case in cluster_cases if not case.get("included_in_calibration", False)]
    wins = sum(1 for delta in deltas if delta > 0)
    losses = sum(1 for delta in deltas if delta < 0)
    ties = sum(1 for delta in deltas if delta == 0)
    median_constraint_delta = None
    if deltas:
        median_constraint_delta = float(np.median(np.array(deltas, dtype=np.float64)))
    rescue_successes = sum(
        1
        for case in selected_cases
        if float(dict(case.get("base_analysis", {})).get("constraint_score") or 0.0) == 0.0
        and float(dict(case.get("anchor_analysis", {})).get("constraint_score") or 0.0) == 1.0
    )
    rescue_rate = None
    if selected_cases:
        rescue_rate = float(rescue_successes / len(selected_cases))
    return {
        "n_total": len(cluster_cases),
        "n_selected": len(selected_cases),
        "mean_constraint_delta": float(sum(deltas) / len(deltas)) if deltas else None,
        "median_constraint_delta": median_constraint_delta,
        "mean_drift_rate": float(sum(drifts) / len(drifts)) if drifts else None,
        "r1_reference_range": [float(min(r1_values)), float(max(r1_values))] if r1_values else [None, None],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "rescue_rate": rescue_rate,
        "excluded_case_names": excluded_case_names,
    }


def build_calibration_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    reference_layers = (
        dict(cases[0].get("reference_layers", {}))
        if cases and isinstance(cases[0].get("reference_layers", {}), dict)
        else {}
    )
    by_cluster_all_cases = {
        cluster: _cluster_summary(cases, cluster, selector="all")
        for cluster in ("mature", "template", "flat")
    }
    by_cluster_clean_base = {
        cluster: _cluster_summary(cases, cluster, selector="included_in_calibration")
        for cluster in ("mature", "template", "flat")
    }
    by_cluster_degenerate_base = {
        cluster: _cluster_summary(cases, cluster, selector="base_degenerate")
        for cluster in ("mature", "template", "flat")
    }
    flat_mean = by_cluster_clean_base["flat"]["mean_constraint_delta"]
    template_mean = by_cluster_clean_base["template"]["mean_constraint_delta"]
    mature_mean = by_cluster_clean_base["mature"]["mean_constraint_delta"]
    clean_base_observed_separation = False
    if flat_mean is not None and template_mean is not None and mature_mean is not None:
        clean_base_observed_separation = bool(flat_mean < min(template_mean, mature_mean))
    return {
        "n_total_cases": len(cases),
        "n_included_cases": sum(1 for case in cases if case.get("included_in_calibration", False)),
        "excluded_base_degenerate_case_names": [
            case["name"] for case in cases if case.get("base_degenerate", False)
        ],
        "by_cluster": by_cluster_clean_base,
        "by_cluster_all_cases": by_cluster_all_cases,
        "by_cluster_clean_base": by_cluster_clean_base,
        "by_cluster_degenerate_base": by_cluster_degenerate_base,
        "reference_layers": reference_layers,
        "threshold_candidates": {
            "r1_reference_mature_threshold": 0.65,
            "delta_template_pair_threshold": 0.08,
            "observed_separation": clean_base_observed_separation,
            "clean_base_observed_separation": clean_base_observed_separation,
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def _constraint_value(case: dict[str, Any], branch: str) -> float:
    analysis = dict(case.get(f"{branch}_analysis", {}))
    value = analysis.get("constraint_score")
    if value is None:
        return 0.0
    return float(value)


def _policy_choice(case: dict[str, Any], policy_name: str) -> tuple[str, float]:
    base_score = _constraint_value(case, "base")
    anchor_score = _constraint_value(case, "anchor")
    if policy_name == "always_base":
        return "base", base_score
    if policy_name == "always_anchor":
        return "anchor", anchor_score
    if policy_name == "failure_gated_any":
        if base_score < 1.0:
            return "anchor", anchor_score
        return "base", base_score
    if policy_name == "flat_anchor":
        if case["anchor_cluster"] == "flat":
            return "anchor", anchor_score
        return "base", base_score
    if policy_name == "flat_failure_gated":
        if case["anchor_cluster"] == "flat" and (
            bool(case.get("base_degenerate", False)) or base_score < 1.0
        ):
            return "anchor", anchor_score
        return "base", base_score
    raise ValueError(f"unknown policy: {policy_name}")


def build_policy_simulation(cases: list[dict[str, Any]]) -> dict[str, Any]:
    policy_names = [
        "always_base",
        "always_anchor",
        "failure_gated_any",
        "flat_anchor",
        "flat_failure_gated",
    ]
    subsets = {
        "all_cases": list(cases),
        "clean_base": [case for case in cases if not bool(case.get("base_degenerate", False))],
        "degenerate_base": [case for case in cases if bool(case.get("base_degenerate", False))],
    }
    summary: dict[str, Any] = {}
    for subset_name, subset_cases in subsets.items():
        base_scores = [_constraint_value(case, "base") for case in subset_cases]
        baseline_mean = float(sum(base_scores) / len(base_scores)) if base_scores else None
        subset_result: dict[str, Any] = {}
        for policy_name in policy_names:
            chosen_scores: list[float] = []
            anchor_pick_count = 0
            wins_over_base = 0
            losses_vs_base = 0
            ties_vs_base = 0
            for case in subset_cases:
                branch, chosen_score = _policy_choice(case, policy_name)
                base_score = _constraint_value(case, "base")
                chosen_scores.append(chosen_score)
                if branch == "anchor":
                    anchor_pick_count += 1
                if chosen_score > base_score:
                    wins_over_base += 1
                elif chosen_score < base_score:
                    losses_vs_base += 1
                else:
                    ties_vs_base += 1
            mean_constraint = float(sum(chosen_scores) / len(chosen_scores)) if chosen_scores else None
            delta_vs_base = None
            if mean_constraint is not None and baseline_mean is not None:
                delta_vs_base = float(mean_constraint - baseline_mean)
            subset_result[policy_name] = {
                "n_cases": len(subset_cases),
                "mean_constraint_score": mean_constraint,
                "delta_vs_always_base": delta_vs_base,
                "anchor_pick_count": anchor_pick_count,
                "anchor_pick_rate": (
                    float(anchor_pick_count / len(subset_cases)) if subset_cases else None
                ),
                "wins_over_base": wins_over_base,
                "losses_vs_base": losses_vs_base,
                "ties_vs_base": ties_vs_base,
            }
        summary[subset_name] = subset_result
    return summary


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    cases: list[dict[str, Any]],
    calibration: dict[str, Any],
    policy_simulation: dict[str, Any],
) -> str:
    reference_layers = dict(calibration.get("reference_layers", {}))
    mature_layer = reference_layers.get("mature_layer")
    template_prev_layer = reference_layers.get("template_prev_layer")
    template_curr_layer = reference_layers.get("template_curr_layer")
    slope_start_layer = reference_layers.get("slope_start_layer")
    slope_end_layer = reference_layers.get("slope_end_layer")
    r1_label = f"r1@L{mature_layer}" if mature_layer is not None else "r1@ref"
    delta_label = (
        f"delta_L{template_prev_layer}→L{template_curr_layer}"
        if template_prev_layer is not None and template_curr_layer is not None
        else "delta_template_pair"
    )
    slope_label = (
        f"slope_L{slope_start_layer}-L{slope_end_layer}"
        if slope_start_layer is not None and slope_end_layer is not None
        else "slope_tail_window"
    )
    cluster_counts = {
        cluster: sum(1 for case in cases if case["anchor_cluster"] == cluster)
        for cluster in ("mature", "template", "flat")
    }
    lines = [
        "# Qwen Geometry Generation Calibration",
        "",
        "## Summary",
        "",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        f"- n_cases: `{len(cases)}`",
        f"- n_included_in_calibration: `{calibration['n_included_cases']}`",
            f"- mature: `{cluster_counts['mature']}`",
            f"- template: `{cluster_counts['template']}`",
            f"- flat: `{cluster_counts['flat']}`",
            f"- excluded_base_degenerate_cases: `{len(calibration['excluded_base_degenerate_case_names'])}`",
            f"- reference_layers: `{reference_layers}`",
        "",
        "## Per-case table",
        "",
        f"| name | cluster | {r1_label} | {delta_label} | {slope_label} | base_constraint | anchor_constraint | constraint_delta | base_degenerate | drift_detected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case["name"],
                    case["anchor_cluster"],
                    _fmt(case["r1_reference"]),
                    _fmt(case["delta_template_pair"]),
                    _fmt(case["slope_tail_window"]),
                    _fmt(case["base_analysis"]["constraint_score"]),
                    _fmt(case["anchor_analysis"]["constraint_score"]),
                    _fmt(case["constraint_delta"]),
                    str(case["base_degenerate"]),
                    str(case["anchor_analysis"]["drift_detected"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Calibration summary",
            "",
            f"| cluster | n_total | n_selected | mean_constraint_delta | median_constraint_delta | mean_drift_rate | rescue_rate | wins | losses | ties | {r1_label}_range |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cluster, summary in calibration["by_cluster_clean_base"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    cluster,
                    str(summary["n_total"]),
                    str(summary["n_selected"]),
                    _fmt(summary["mean_constraint_delta"]),
                    _fmt(summary["median_constraint_delta"]),
                    _fmt(summary["mean_drift_rate"]),
                    _fmt(summary["rescue_rate"]),
                    str(summary["wins"]),
                    str(summary["losses"]),
                    str(summary["ties"]),
                    f"[{_fmt(summary['r1_reference_range'][0])}, {_fmt(summary['r1_reference_range'][1])}]",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Degenerate-base rescue summary",
            "",
            "| cluster | n_total | n_selected | mean_constraint_delta | median_constraint_delta | rescue_rate | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cluster, summary in calibration["by_cluster_degenerate_base"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    cluster,
                    str(summary["n_total"]),
                    str(summary["n_selected"]),
                    _fmt(summary["mean_constraint_delta"]),
                    _fmt(summary["median_constraint_delta"]),
                    _fmt(summary["rescue_rate"]),
                    str(summary["wins"]),
                    str(summary["losses"]),
                    str(summary["ties"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Policy simulation",
            "",
        ]
    )
    for subset_name, subset_summary in policy_simulation.items():
        lines.extend(
            [
                f"### {subset_name}",
                "",
                "| policy | n_cases | mean_constraint_score | delta_vs_always_base | anchor_pick_rate | wins_over_base | losses_vs_base | ties_vs_base |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for policy_name, policy_stats in subset_summary.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        policy_name,
                        str(policy_stats["n_cases"]),
                        _fmt(policy_stats["mean_constraint_score"]),
                        _fmt(policy_stats["delta_vs_always_base"]),
                        _fmt(policy_stats["anchor_pick_rate"]),
                        str(policy_stats["wins_over_base"]),
                        str(policy_stats["losses_vs_base"]),
                        str(policy_stats["ties_vs_base"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            f"- excluded_base_degenerate_case_names: `{calibration['excluded_base_degenerate_case_names']}`",
            "",
            f"- clean_base_observed_separation: `{calibration['threshold_candidates']['clean_base_observed_separation']}`",
            "",
            "## Conclusion",
            "",
            f"Current data {'support' if calibration['threshold_candidates']['clean_base_observed_separation'] else 'do not support'} the thresholds `{r1_label} > 0.65` and `{delta_label} > 0.08` as a clean routing split on non-degenerate-base cases; degenerate-base rescue cases are reported separately.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_json: Path,
    output_md: Path,
    payload: dict[str, Any],
    report: str,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run geometry + generation calibration for Qwen anchors.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_geometry_generation_calibration.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_geometry_generation_calibration.md",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.10,
        anchor_revision_threshold=0.35,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.50,
    )
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model,
        cfg=cfg,
        device=args.device,
        torch_dtype=torch.float16 if "cuda" in args.device else None,
        low_cpu_mem_usage=True,
    )
    overlay.eval()
    device = torch.device(args.device)
    probe_layers, reference_layers = resolve_geometry_probe_spec(overlay)

    cases = make_qwen_anchor_geometry_cases()
    if args.case_name:
        cases = [case for case in cases if case.name == args.case_name]
    if args.limit is not None:
        cases = cases[: max(int(args.limit), 0)]
    records: list[dict[str, Any]] = []
    if args.resume and args.output_json.exists():
        existing_payload = json.loads(args.output_json.read_text(encoding="utf-8"))
        existing_cases = existing_payload.get("cases", [])
        if isinstance(existing_cases, list):
            records = [case for case in existing_cases if isinstance(case, dict)]
            processed_names = {str(case.get("name")) for case in records if case.get("name")}
            cases = [case for case in cases if case.name not in processed_names]
            print(f"RESUME loaded {len(records)} cases from {args.output_json}")
    for case in cases:
        record = analyze_case(
            overlay=overlay,
            case=case,
            probe_layers=probe_layers,
            reference_layers=reference_layers,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            conflict_threshold=DEFAULT_CONFLICT_THRESHOLD,
            bias_scale=DEFAULT_BIAS_SCALE,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
            no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
            max_bias_gate_sum=DEFAULT_MAX_BIAS_GATE_SUM,
            entropy_top_k=DEFAULT_ENTROPY_TOP_K,
            entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
            entropy_slope=DEFAULT_ENTROPY_SLOPE,
            pressure_threshold=DEFAULT_PRESSURE_THRESHOLD,
            pressure_slope=DEFAULT_PRESSURE_SLOPE,
            pressure_rescue_floor=DEFAULT_PRESSURE_RESCUE_FLOOR,
            device=device,
        )
        if record is not None:
            records.append(record)
            calibration = build_calibration_summary(records)
            policy_simulation = build_policy_simulation(records)
            payload = {
                "metadata": {
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "model_name": args.model,
                    "device": args.device,
                    "max_length": args.max_length,
                    "max_new_tokens": args.max_new_tokens,
                    "probe_layers": probe_layers,
                    "reference_layers": reference_layers,
                    "seed": args.seed,
                },
                "cases": records,
                "calibration": calibration,
                "policy_simulation": policy_simulation,
            }
            report = build_markdown_report(
                model_name=args.model,
                device=args.device,
                cases=records,
                calibration=calibration,
                policy_simulation=policy_simulation,
            )
            write_outputs(
                output_json=args.output_json,
                output_md=args.output_md,
                payload=payload,
                report=report,
            )

    calibration = build_calibration_summary(records)
    policy_simulation = build_policy_simulation(records)
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": args.model,
            "device": args.device,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "probe_layers": probe_layers,
            "reference_layers": reference_layers,
            "seed": args.seed,
        },
        "cases": records,
        "calibration": calibration,
        "policy_simulation": policy_simulation,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        cases=records,
        calibration=calibration,
        policy_simulation=policy_simulation,
    )

    write_outputs(
        output_json=args.output_json,
        output_md=args.output_md,
        payload=payload,
        report=report,
    )

    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
