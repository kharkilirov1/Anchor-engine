from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
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
    list_model_layers,
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
DEFAULT_SEARCH_TAIL_LAYER_COUNT = 12
DEFAULT_TOP_SEARCH_CANDIDATES = 10
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


def resolve_geometry_probe_layers(
    overlay: QwenAnchorOverlay,
) -> tuple[list[int], list[int]]:
    num_hidden_layers = int(getattr(overlay, "model_num_hidden_layers", 0))
    if num_hidden_layers <= 0:
        raise ValueError("model must expose a positive num_hidden_layers value")
    probe_layers = list_model_layers(num_hidden_layers)
    search_layers = select_tail_probe_layers(
        num_hidden_layers=num_hidden_layers,
        count=min(DEFAULT_SEARCH_TAIL_LAYER_COUNT, num_hidden_layers),
    )
    return probe_layers, search_layers

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
    coherence_profile: dict[str, float | None] = {}
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
        coherence_profile[str(layer)] = _to_scalar(metrics.get("adjacent_cosine_coherence"))
        layer_metrics[str(layer)] = {key: _to_scalar(value) for key, value in metrics.items()}

    return {
        "span_match": asdict(span_match),
        "matched_token_ids": [int(token_id) for token_id in matched_token_ids],
        "raw_tokens": raw_tokens,
        "decoded_tokens": decoded_tokens,
        "first_token_has_leading_whitespace": token_has_leading_whitespace(raw_tokens[0], decoded_tokens[0]) if raw_tokens else None,
        "rank1_profile": rank1_profile,
        "tortuosity_profile": tortuosity_profile,
        "coherence_profile": coherence_profile,
        "layer_metrics": layer_metrics,
        "probe_layers": [int(layer) for layer in probe_layers],
    }


def compute_case_reference_features(
    case: dict[str, Any],
    reference_layers: dict[str, int],
) -> dict[str, Any]:
    rank1_profile = dict(case.get("rank1_profile", {}))
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
        (layer, rank1_profile.get(str(layer)))
        for layer in range(slope_start_layer, slope_end_layer + 1)
        if rank1_profile.get(str(layer)) is not None
    ]
    if len(slope_points) >= 2:
        xs = np.array([point[0] for point in slope_points], dtype=np.float64)
        ys = np.array([float(point[1]) for point in slope_points], dtype=np.float64)
        slope_tail_window = float(np.polyfit(xs, ys, deg=1)[0])

    return {
        "reference_layers": {key: int(value) for key, value in reference_layers.items()},
        "r1_reference": _to_scalar(r1_reference),
        "delta_template_pair": _to_scalar(delta_template_pair),
        "slope_tail_window": _to_scalar(slope_tail_window),
    }


def classify_anchor_cluster(
    *,
    r1_reference: float | int | None,
    delta_template_pair: float | int | None,
    mature_threshold: float,
    template_threshold: float,
) -> str:
    if r1_reference is not None and float(r1_reference) >= float(mature_threshold):
        return "mature"
    if delta_template_pair is not None and float(delta_template_pair) >= float(template_threshold):
        return "template"
    return "flat"


def apply_cluster_configuration(
    cases: list[dict[str, Any]],
    *,
    reference_layers: dict[str, int],
    mature_threshold: float,
    template_threshold: float,
) -> list[dict[str, Any]]:
    enriched_cases: list[dict[str, Any]] = []
    for case in cases:
        features = compute_case_reference_features(case, reference_layers=reference_layers)
        cluster = classify_anchor_cluster(
            r1_reference=features["r1_reference"],
            delta_template_pair=features["delta_template_pair"],
            mature_threshold=mature_threshold,
            template_threshold=template_threshold,
        )
        enriched_case = dict(case)
        enriched_case.update(features)
        enriched_case["anchor_cluster"] = cluster
        enriched_case["mature_threshold"] = float(mature_threshold)
        enriched_case["template_threshold"] = float(template_threshold)
        enriched_cases.append(enriched_case)
    return enriched_cases


def _cluster_entropy(counts: dict[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        if value <= 0:
            continue
        prob = float(value) / total
        entropy -= prob * math.log(prob + 1e-12)
    return float(entropy / math.log(3.0))


def _candidate_thresholds(values: list[float]) -> list[float]:
    filtered = sorted({float(value) for value in values if value is not None and math.isfinite(float(value))})
    if not filtered:
        return [0.0]
    candidates = {filtered[0], filtered[-1] + 1e-6}
    for left, right in zip(filtered, filtered[1:]):
        candidates.add((left + right) / 2.0)
    quantile_points = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
    array = np.array(filtered, dtype=np.float64)
    for q in quantile_points:
        candidates.add(float(np.quantile(array, q)))
    return sorted(candidates)


def iter_reference_layer_candidates(search_layers: list[int]) -> list[dict[str, int]]:
    if len(search_layers) < 4:
        return [build_tail_reference_layers(search_layers)]
    candidates: list[dict[str, int]] = []
    min_layer = int(search_layers[0])
    mature_candidates = [int(layer) for layer in search_layers[:-2]]
    for mature_layer in mature_candidates:
        template_pairs = [
            (int(prev_layer), int(curr_layer))
            for prev_layer, curr_layer in zip(search_layers, search_layers[1:])
            if int(curr_layer) > mature_layer
        ]
        for template_prev_layer, template_curr_layer in template_pairs:
            slope_start_layer = max(min_layer, mature_layer - 6)
            candidates.append(
                {
                    "slope_start_layer": slope_start_layer,
                    "slope_end_layer": mature_layer,
                    "mature_layer": mature_layer,
                    "template_prev_layer": template_prev_layer,
                    "template_curr_layer": template_curr_layer,
                }
            )
    deduped: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for candidate in candidates:
        key = (
            candidate["slope_start_layer"],
            candidate["slope_end_layer"],
            candidate["mature_layer"],
            candidate["template_prev_layer"],
            candidate["template_curr_layer"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _geometry_margin(enriched_cases: list[dict[str, Any]]) -> float:
    by_cluster: dict[str, list[dict[str, Any]]] = {
        cluster: [case for case in enriched_cases if case.get("anchor_cluster") == cluster]
        for cluster in ("mature", "template", "flat")
    }
    margin = 0.0
    if by_cluster["mature"]:
        mature_mean = float(np.mean([float(case["r1_reference"]) for case in by_cluster["mature"] if case.get("r1_reference") is not None]))
        others = [
            float(case["r1_reference"])
            for cluster in ("template", "flat")
            for case in by_cluster[cluster]
            if case.get("r1_reference") is not None
        ]
        if others:
            margin += max(0.0, mature_mean - max(others))
    if by_cluster["template"]:
        template_mean = float(np.mean([float(case["delta_template_pair"]) for case in by_cluster["template"] if case.get("delta_template_pair") is not None]))
        others = [
            float(case["delta_template_pair"])
            for cluster in ("mature", "flat")
            for case in by_cluster[cluster]
            if case.get("delta_template_pair") is not None
        ]
        if others:
            margin += max(0.0, template_mean - max(others))
    return float(margin)


def score_cluster_configuration(
    enriched_cases: list[dict[str, Any]],
    calibration: dict[str, Any],
    policy_simulation: dict[str, Any],
) -> float:
    all_flat_failure = dict(policy_simulation.get("all_cases", {})).get("flat_failure_gated", {})
    clean_flat_failure = dict(policy_simulation.get("clean_base", {})).get("flat_failure_gated", {})
    delta_all = float(all_flat_failure.get("delta_vs_always_base") or 0.0)
    delta_clean = float(clean_flat_failure.get("delta_vs_always_base") or 0.0)
    wins = int(all_flat_failure.get("wins_over_base") or 0)
    losses = int(all_flat_failure.get("losses_vs_base") or 0)
    counts = {
        cluster: sum(1 for case in enriched_cases if case.get("anchor_cluster") == cluster)
        for cluster in ("mature", "template", "flat")
    }
    nonempty_clusters = sum(1 for value in counts.values() if value > 0)
    entropy = _cluster_entropy(counts)
    geometry_margin = _geometry_margin(enriched_cases)
    flat_rescue = float(dict(calibration.get("by_cluster_all_cases", {})).get("flat", {}).get("rescue_rate") or 0.0)
    template_losses = int(dict(calibration.get("by_cluster_clean_base", {})).get("template", {}).get("losses") or 0)
    mature_losses = int(dict(calibration.get("by_cluster_clean_base", {})).get("mature", {}).get("losses") or 0)
    return float(
        100.0 * delta_all
        + 50.0 * delta_clean
        + 8.0 * wins
        - 12.0 * losses
        + 6.0 * flat_rescue
        + 2.0 * nonempty_clusters
        + 3.0 * entropy
        + 20.0 * geometry_margin
        - 3.0 * template_losses
        - 2.0 * mature_losses
    )


def search_reference_and_thresholds(
    cases: list[dict[str, Any]],
    *,
    search_layers: list[int],
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    top_candidates: list[dict[str, Any]] = []
    reference_candidates = iter_reference_layer_candidates(search_layers)
    for reference_layers in reference_candidates:
        feature_views = [compute_case_reference_features(case, reference_layers) for case in cases]
        mature_thresholds = _candidate_thresholds(
            [view["r1_reference"] for view in feature_views if view.get("r1_reference") is not None]
        )
        template_thresholds = _candidate_thresholds(
            [view["delta_template_pair"] for view in feature_views if view.get("delta_template_pair") is not None]
        )
        for mature_threshold in mature_thresholds:
            for template_threshold in template_thresholds:
                enriched_cases = apply_cluster_configuration(
                    cases,
                    reference_layers=reference_layers,
                    mature_threshold=float(mature_threshold),
                    template_threshold=float(template_threshold),
                )
                calibration = build_calibration_summary(
                    enriched_cases,
                    reference_layers=reference_layers,
                    thresholds={
                        "mature_r1_threshold": float(mature_threshold),
                        "template_delta_threshold": float(template_threshold),
                    },
                )
                policy_simulation = build_policy_simulation(enriched_cases)
                score = score_cluster_configuration(
                    enriched_cases=enriched_cases,
                    calibration=calibration,
                    policy_simulation=policy_simulation,
                )
                counts = {
                    cluster: sum(1 for case in enriched_cases if case.get("anchor_cluster") == cluster)
                    for cluster in ("mature", "template", "flat")
                }
                candidate = {
                    "score": float(score),
                    "reference_layers": {key: int(value) for key, value in reference_layers.items()},
                    "thresholds": {
                        "mature_r1_threshold": float(mature_threshold),
                        "template_delta_threshold": float(template_threshold),
                    },
                    "cluster_counts": counts,
                    "policy_flat_failure_gated_all_cases": dict(policy_simulation.get("all_cases", {})).get("flat_failure_gated", {}),
                    "policy_flat_failure_gated_clean_base": dict(policy_simulation.get("clean_base", {})).get("flat_failure_gated", {}),
                    "clean_base_observed_separation": bool(dict(calibration.get("threshold_candidates", {})).get("clean_base_observed_separation", False)),
                    "geometry_margin": _geometry_margin(enriched_cases),
                }
                top_candidates.append(candidate)
                if best_result is None or float(candidate["score"]) > float(best_result["candidate"]["score"]):
                    best_result = {
                        "candidate": candidate,
                        "cases": enriched_cases,
                        "calibration": calibration,
                        "policy_simulation": policy_simulation,
                    }
    if best_result is None:
        raise ValueError("failed to search reference layers and thresholds")
    ranked = sorted(top_candidates, key=lambda item: float(item["score"]), reverse=True)
    best_result["search_summary"] = {
        "n_reference_candidates": len(reference_candidates),
        "n_total_candidates": len(top_candidates),
        "top_candidates": ranked[:DEFAULT_TOP_SEARCH_CANDIDATES],
        "search_layers": [int(layer) for layer in search_layers],
        "best_candidate": best_result["candidate"],
    }
    return best_result


def analyze_case(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    *,
    probe_layers: list[int],
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

    print(f"DONE {case.name}: geometry_ready")
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
        "coherence_profile": geometry["coherence_profile"],
        "probe_layers": geometry["probe_layers"],
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


def build_calibration_summary(
    cases: list[dict[str, Any]],
    *,
    reference_layers: dict[str, int] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    resolved_reference_layers = (
        dict(reference_layers)
        if isinstance(reference_layers, dict)
        else (
            dict(cases[0].get("reference_layers", {}))
            if cases and isinstance(cases[0].get("reference_layers", {}), dict)
            else {}
        )
    )
    resolved_thresholds = dict(thresholds) if isinstance(thresholds, dict) else {}
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
        "reference_layers": resolved_reference_layers,
        "threshold_candidates": {
            "r1_reference_mature_threshold": resolved_thresholds.get("mature_r1_threshold"),
            "delta_template_pair_threshold": resolved_thresholds.get("template_delta_threshold"),
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
    search: dict[str, Any],
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
    threshold_candidates = dict(calibration.get("threshold_candidates", {}))
    mature_threshold = threshold_candidates.get("r1_reference_mature_threshold")
    template_threshold = threshold_candidates.get("delta_template_pair_threshold")
    cluster_counts = {
        cluster: sum(1 for case in cases if case["anchor_cluster"] == cluster)
        for cluster in ("mature", "template", "flat")
    }
    top_candidates = list(search.get("top_candidates", []))
    lines = [
        "# Qwen Geometry Generation Calibration",
        "",
        "## Summary",
        "",
        f"- Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        f"- n_cases: `{len(cases)}`",
        f"- n_included_in_calibration: `{calibration['n_included_cases']}`",
        f"- mature: `{cluster_counts['mature']}`",
        f"- template: `{cluster_counts['template']}`",
        f"- flat: `{cluster_counts['flat']}`",
        f"- excluded_base_degenerate_cases: `{len(calibration['excluded_base_degenerate_case_names'])}`",
        f"- search_layers: `{search.get('search_layers', [])}`",
        f"- searched_reference_layers: `{reference_layers}`",
        f"- searched_thresholds: `{{'mature_r1_threshold': {mature_threshold}, 'template_delta_threshold': {template_threshold}}}`",
        f"- n_reference_candidates: `{search.get('n_reference_candidates', 0)}`",
        f"- n_total_candidates: `{search.get('n_total_candidates', 0)}`",
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
            "## Search summary",
            "",
            "| rank | score | reference_layers | thresholds | cluster_counts | flat_failure_gated_all_cases |",
            "| ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for idx, candidate in enumerate(top_candidates, start=1):
        flat_failure = dict(candidate.get("policy_flat_failure_gated_all_cases", {}))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    _fmt(candidate.get("score")),
                    f"`{candidate.get('reference_layers', {})}`",
                    f"`{candidate.get('thresholds', {})}`",
                    f"`{candidate.get('cluster_counts', {})}`",
                    f"`{{'delta_vs_base': {flat_failure.get('delta_vs_always_base')}, 'wins': {flat_failure.get('wins_over_base')}, 'losses': {flat_failure.get('losses_vs_base')}}}`",
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
            (
                f"Current data {'support' if calibration['threshold_candidates']['clean_base_observed_separation'] else 'do not support'} "
                f"the searched split `{r1_label} >= {_fmt(mature_threshold)}` and "
                f"`{delta_label} >= {_fmt(template_threshold)}` as a clean routing split on non-degenerate-base cases."
            ),
            "",
            (
                "This report searched reference layers and crystallization thresholds on the current host model; "
                "the values above are model-specific and should not be interpreted as direct transfers from older Qwen runs."
            ),
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
    probe_layers, search_layers = resolve_geometry_probe_layers(overlay)

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
    if not cases and not records:
        raise ValueError("no cases selected for calibration")
    for case in cases:
        record = analyze_case(
            overlay=overlay,
            case=case,
            probe_layers=probe_layers,
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
            search_result = search_reference_and_thresholds(records, search_layers=search_layers)
            enriched_cases = search_result["cases"]
            calibration = search_result["calibration"]
            policy_simulation = search_result["policy_simulation"]
            best_candidate = dict(search_result["candidate"])
            payload = {
                "metadata": {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model_name": args.model,
                    "device": args.device,
                    "max_length": args.max_length,
                    "max_new_tokens": args.max_new_tokens,
                    "probe_layers": probe_layers,
                    "search_layers": search_layers,
                    "reference_layers": best_candidate["reference_layers"],
                    "seed": args.seed,
                },
                "cases": enriched_cases,
                "calibration": calibration,
                "policy_simulation": policy_simulation,
                "search": search_result["search_summary"],
            }
            report = build_markdown_report(
                model_name=args.model,
                device=args.device,
                cases=enriched_cases,
                calibration=calibration,
                policy_simulation=policy_simulation,
                search=search_result["search_summary"],
            )
            write_outputs(
                output_json=args.output_json,
                output_md=args.output_md,
                payload=payload,
                report=report,
            )

    search_result = search_reference_and_thresholds(records, search_layers=search_layers)
    enriched_cases = search_result["cases"]
    calibration = search_result["calibration"]
    policy_simulation = search_result["policy_simulation"]
    best_candidate = dict(search_result["candidate"])
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": args.model,
            "device": args.device,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "probe_layers": probe_layers,
            "search_layers": search_layers,
            "reference_layers": best_candidate["reference_layers"],
            "seed": args.seed,
        },
        "cases": enriched_cases,
        "calibration": calibration,
        "policy_simulation": policy_simulation,
        "search": search_result["search_summary"],
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        cases=enriched_cases,
        calibration=calibration,
        policy_simulation=policy_simulation,
        search=search_result["search_summary"],
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
