from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import json
import math
from pathlib import Path
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
    compute_geometry_metrics,
    extract_delta_vectors,
    match_anchor_span,
    decode_token_pieces,
    decode_token_surfaces,
    token_has_leading_whitespace,
)
from src.data.qwen_anchor_geometry_cases import (
    make_qwen_anchor_geometry_cases,
    QwenAnchorGeometryCase,
)


PROBE_LAYERS = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
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

KEYWORD_MAP: dict[str, dict[str, list[str]]] = {
    "strictly_vegan_meal_plan_policy": {
        "positive": ["vegan", "plant-based", "tofu", "lentil", "chickpea", "beans", "vegetable", "mushroom"],
        "negative": ["egg", "eggs", "cheese", "butter", "milk", "cream", "meat", "chicken", "beef"],
    },
    "async_fastapi_service_architecture_policy": {
        "positive": ["async", "await", "FastAPI", "router", "endpoint", "dependency", "asyncio"],
        "negative": ["Flask", "Django", "sync", "blocking", "thread", "subprocess"],
    },
    "json_only_response_format_policy": {
        "positive": ["json", "JSON", "{", "}", "key", "value", "format"],
        "negative": ["markdown", "plain text", "prose", "sorry", "I cannot", "Here is"],
    },
    "proof_by_contradiction_reasoning_steps": {
        "positive": ["assume", "contradiction", "suppose", "therefore", "absurd", "QED", "proof"],
        "negative": ["example", "for instance", "because", "simply", "just", "obviously"],
    },
    "binary_search_update_loop_procedure": {
        "positive": ["mid", "low", "high", "left", "right", "while", "binary", "O(log"],
        "negative": ["for i", "linear", "scan", "iterate", "brute"],
    },
    "dependency_injection_request_flow_sequence": {
        "positive": ["inject", "dependency", "container", "resolve", "provider", "interface"],
        "negative": ["global", "singleton", "import", "hardcode", "direct instantiation"],
    },
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


def _to_scalar(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if not math.isfinite(float(value)):
        return None
    return float(value)


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
    quality_score = lexical_score - 1.5 * float(sum(negative_hits.values()))
    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "protected_negative_hits": protected_negative_hits,
        "positive_total": int(sum(positive_hits.values())),
        "negative_total": int(sum(negative_hits.values())),
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
    for layer in PROBE_LAYERS:
        delta_vectors = extract_delta_vectors(
            hidden_states[layer + 1][0],
            span_match.token_start,
            span_match.token_end,
        )
        metrics = compute_geometry_metrics(delta_vectors)
        rank1_profile[str(layer)] = _to_scalar(metrics.get("rank1_explained_variance"))
        tortuosity_profile[str(layer)] = _to_scalar(metrics.get("path_tortuosity"))
        layer_metrics[str(layer)] = {key: _to_scalar(value) for key, value in metrics.items()}

    r1_at_24 = rank1_profile.get("24")
    r1_l26 = rank1_profile.get("26")
    r1_l27 = rank1_profile.get("27")
    delta_l26_l27 = None
    if r1_l26 is not None and r1_l27 is not None:
        delta_l26_l27 = float(r1_l27 - r1_l26)

    slope_l18_l24 = None
    slope_points = [
        (layer, rank1_profile[str(layer)])
        for layer in range(18, 25)
        if rank1_profile.get(str(layer)) is not None
    ]
    if len(slope_points) >= 2:
        xs = np.array([point[0] for point in slope_points], dtype=np.float64)
        ys = np.array([float(point[1]) for point in slope_points], dtype=np.float64)
        slope_l18_l24 = float(np.polyfit(xs, ys, deg=1)[0])

    if r1_at_24 is not None and float(r1_at_24) > 0.65:
        anchor_cluster = "mature"
    elif delta_l26_l27 is not None and float(delta_l26_l27) > 0.08:
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
        "r1_at_24": _to_scalar(r1_at_24),
        "delta_l26_l27": _to_scalar(delta_l26_l27),
        "slope_l18_l24": _to_scalar(slope_l18_l24),
        "anchor_cluster": anchor_cluster,
    }


def analyze_case(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    *,
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
    )
    if geometry is None:
        print(f"SKIP {case.name}: span not matched")
        return None

    keyword_spec = KEYWORD_MAP.get(case.anchor_group)
    positive_keywords = keyword_spec["positive"] if keyword_spec is not None else []
    negative_keywords = keyword_spec["negative"] if keyword_spec is not None else []

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
            "quality_score": None,
            "drift_detected": False,
        }
        anchor_analysis = {
            "positive_total": 0,
            "negative_total": 0,
            "quality_score": None,
            "drift_detected": False,
        }
        constraint_delta = None
    else:
        base_raw = analyze_keywords(base_continuation, positive_keywords=positive_keywords, negative_keywords=negative_keywords)
        anchor_raw = analyze_keywords(anchor_continuation, positive_keywords=positive_keywords, negative_keywords=negative_keywords)
        base_analysis = {
            "positive_total": int(base_raw["positive_total"]),
            "negative_total": int(base_raw["negative_total"]),
            "quality_score": float(base_raw["quality_score"]),
            "drift_detected": bool(base_raw["negative_total"] > 0),
        }
        anchor_analysis = {
            "positive_total": int(anchor_raw["positive_total"]),
            "negative_total": int(anchor_raw["negative_total"]),
            "quality_score": float(anchor_raw["quality_score"]),
            "drift_detected": bool(anchor_raw["negative_total"] > 0),
        }
        constraint_delta = float(anchor_analysis["quality_score"] - base_analysis["quality_score"])

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
        "r1_at_24": geometry["r1_at_24"],
        "delta_l26_l27": geometry["delta_l26_l27"],
        "slope_l18_l24": geometry["slope_l18_l24"],
        "anchor_cluster": geometry["anchor_cluster"],
        "base_continuation": base_continuation,
        "anchor_continuation": anchor_continuation,
        "base_analysis": base_analysis,
        "anchor_analysis": anchor_analysis,
        "constraint_delta": _to_scalar(constraint_delta),
    }


def _cluster_summary(cases: list[dict[str, Any]], cluster: str) -> dict[str, Any]:
    cluster_cases = [case for case in cases if case["anchor_cluster"] == cluster]
    deltas = [float(case["constraint_delta"]) for case in cluster_cases if case["constraint_delta"] is not None]
    drifts = [1.0 if case["anchor_analysis"]["drift_detected"] else 0.0 for case in cluster_cases]
    r1_values = [float(case["r1_at_24"]) for case in cluster_cases if case["r1_at_24"] is not None]
    return {
        "n": len(cluster_cases),
        "mean_constraint_delta": float(sum(deltas) / len(deltas)) if deltas else None,
        "mean_drift_rate": float(sum(drifts) / len(drifts)) if drifts else None,
        "r1_at_24_range": [float(min(r1_values)), float(max(r1_values))] if r1_values else [None, None],
    }


def build_calibration_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_cluster = {cluster: _cluster_summary(cases, cluster) for cluster in ("mature", "template", "flat")}
    flat_mean = by_cluster["flat"]["mean_constraint_delta"]
    template_mean = by_cluster["template"]["mean_constraint_delta"]
    mature_mean = by_cluster["mature"]["mean_constraint_delta"]
    observed_separation = False
    if flat_mean is not None and template_mean is not None and mature_mean is not None:
        observed_separation = bool(flat_mean < min(template_mean, mature_mean))
    return {
        "by_cluster": by_cluster,
        "threshold_candidates": {
            "r1_at_24_mature_threshold": 0.65,
            "delta_l26_l27_template_threshold": 0.08,
            "observed_separation": observed_separation,
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    cases: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> str:
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
        f"- mature: `{cluster_counts['mature']}`",
        f"- template: `{cluster_counts['template']}`",
        f"- flat: `{cluster_counts['flat']}`",
        "",
        "## Per-case table",
        "",
        "| name | cluster | r1@L24 | delta_L26→L27 | slope_L18-L24 | base_quality | anchor_quality | constraint_delta | drift_detected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case["name"],
                    case["anchor_cluster"],
                    _fmt(case["r1_at_24"]),
                    _fmt(case["delta_l26_l27"]),
                    _fmt(case["slope_l18_l24"]),
                    _fmt(case["base_analysis"]["quality_score"]),
                    _fmt(case["anchor_analysis"]["quality_score"]),
                    _fmt(case["constraint_delta"]),
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
            "| cluster | n | mean_constraint_delta | mean_drift_rate | r1_at_24_range |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for cluster, summary in calibration["by_cluster"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    cluster,
                    str(summary["n"]),
                    _fmt(summary["mean_constraint_delta"]),
                    _fmt(summary["mean_drift_rate"]),
                    f"[{_fmt(summary['r1_at_24_range'][0])}, {_fmt(summary['r1_at_24_range'][1])}]",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"- observed_separation: `{calibration['threshold_candidates']['observed_separation']}`",
            "",
            "## Conclusion",
            "",
            f"Current data {'support' if calibration['threshold_candidates']['observed_separation'] else 'do not support'} the thresholds `0.65 / 0.08` as a clean routing split.",
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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
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
            payload = {
                "metadata": {
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "model_name": args.model,
                    "device": args.device,
                    "max_length": args.max_length,
                    "max_new_tokens": args.max_new_tokens,
                    "probe_layers": PROBE_LAYERS,
                    "seed": args.seed,
                },
                "cases": records,
                "calibration": calibration,
            }
            report = build_markdown_report(
                model_name=args.model,
                device=args.device,
                cases=records,
                calibration=calibration,
            )
            write_outputs(
                output_json=args.output_json,
                output_md=args.output_md,
                payload=payload,
                report=report,
            )

    calibration = build_calibration_summary(records)
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": args.model,
            "device": args.device,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "probe_layers": PROBE_LAYERS,
            "seed": args.seed,
        },
        "cases": records,
        "calibration": calibration,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        cases=records,
        calibration=calibration,
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
