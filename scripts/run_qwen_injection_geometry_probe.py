from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import QwenAnchorGeometryCase, make_qwen_anchor_geometry_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import compute_geometry_metrics, extract_delta_vectors, match_anchor_span

FEATURE_METRICS: tuple[str, ...] = (
    "rank1_explained_variance",
    "adjacent_cosine_coherence",
    "path_tortuosity",
    "mean_direction_norm",
)


def select_cases(cases: list[QwenAnchorGeometryCase], group_case_cap: int) -> list[QwenAnchorGeometryCase]:
    if group_case_cap <= 0:
        return list(cases)
    selected: list[QwenAnchorGeometryCase] = []
    seen: dict[str, int] = defaultdict(int)
    for case in cases:
        if seen[case.anchor_group] >= group_case_cap:
            continue
        selected.append(case)
        seen[case.anchor_group] += 1
    return selected


def choose_probe_layers(num_hidden_layers: int) -> list[int]:
    preferred = [4, 5, 6, 7, 8, 10, 11, 24, 25]
    layers = [layer for layer in preferred if 0 <= layer < num_hidden_layers]
    if layers:
        return layers
    return list(range(max(0, num_hidden_layers - min(6, num_hidden_layers)), num_hidden_layers))


def build_injected_prompt(target_case: QwenAnchorGeometryCase, host_case: QwenAnchorGeometryCase) -> str:
    return (
        f"Audit preface includes detached anchor tag: {target_case.anchor_text}. "
        "Treat that tag as an unrelated marker while continuing the original task.\n\n"
        f"{host_case.prompt}"
    )


def _to_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def compute_auc(negative_scores: list[float], positive_scores: list[float]) -> float | None:
    if not negative_scores or not positive_scores:
        return None
    wins = 0.0
    total = 0
    for negative in negative_scores:
        for positive in positive_scores:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / total) if total else None


def extract_feature_vector(
    overlay: QwenAnchorOverlay,
    prompt: str,
    anchor_text: str,
    layers: list[int],
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    if overlay.tokenizer is None:
        return {"status": "error", "note": "tokenizer_missing"}
    encoded = overlay.tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoded.pop("offset_mapping", None)
    offsets = None
    if isinstance(offset_mapping, torch.Tensor):
        offsets = [(int(start), int(end)) for start, end in offset_mapping.squeeze(0).tolist()]
    batch = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    input_ids = [int(token_id) for token_id in batch["input_ids"][0].tolist()]
    span_match = match_anchor_span(
        text=prompt,
        anchor_text=anchor_text,
        input_ids=input_ids,
        tokenizer=overlay.tokenizer,
        offsets=offsets,
    )
    if span_match is None:
        return {"status": "skip", "note": "anchor_span_not_matched"}
    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    feature_vector: list[float] = []
    layer_metrics: dict[str, dict[str, float | None]] = {}
    for layer in layers:
        delta_vectors = extract_delta_vectors(hidden_states[layer + 1][0], span_match.token_start, span_match.token_end)
        metrics = compute_geometry_metrics(delta_vectors)
        layer_key = str(layer)
        layer_metrics[layer_key] = {}
        for metric_name in FEATURE_METRICS:
            metric_value = _to_float(metrics.get(metric_name))
            layer_metrics[layer_key][metric_name] = metric_value
            feature_vector.append(0.0 if metric_value is None else float(metric_value))
    return {
        "status": "ok",
        "span_token_count": int(span_match.token_count),
        "match_method": span_match.match_method,
        "feature_vector": feature_vector,
        "layer_metrics": layer_metrics,
    }


def compute_scale(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    scales: list[float] = []
    for dim in range(dims):
        column = [float(vector[dim]) for vector in vectors]
        mean_value = sum(column) / len(column)
        variance = sum((value - mean_value) ** 2 for value in column) / max(1, len(column))
        scales.append(max(variance ** 0.5, 1e-6))
    return scales


def standardized_l1_distance(vector: list[float], prototype: list[float], scale: list[float]) -> float:
    if not vector or not prototype or not scale:
        return 0.0
    total = 0.0
    for value, proto, denom in zip(vector, prototype, scale):
        total += abs(float(value) - float(proto)) / max(float(denom), 1e-6)
    return float(total / len(vector))


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    out: list[float] = []
    for dim in range(dims):
        out.append(float(sum(float(vector[dim]) for vector in vectors) / len(vectors)))
    return out


def build_host_map(cases: list[QwenAnchorGeometryCase]) -> dict[str, QwenAnchorGeometryCase]:
    by_class: dict[str, list[QwenAnchorGeometryCase]] = defaultdict(list)
    for case in cases:
        by_class[case.anchor_class].append(case)
    host_map: dict[str, QwenAnchorGeometryCase] = {}
    for class_cases in by_class.values():
        ordered = sorted(class_cases, key=lambda item: item.name)
        for index, case in enumerate(ordered):
            host_map[case.name] = ordered[(index + 1) % len(ordered)]
    return host_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--profile", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--group-case-cap", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args()

    device = torch.device(args.device)
    torch.manual_seed(7)

    overlay = QwenAnchorOverlay.from_pretrained(args.model_name, config=TOY_CONFIG)
    overlay.to(device)
    overlay.eval()

    num_hidden_layers = int(overlay.model_num_hidden_layers)
    layers = choose_probe_layers(num_hidden_layers)
    selected_cases = select_cases(
        make_qwen_anchor_geometry_cases(anchor_span_profile=args.profile),
        group_case_cap=args.group_case_cap,
    )
    host_map = build_host_map(selected_cases)

    legit_rows: list[dict[str, Any]] = []
    injected_rows: list[dict[str, Any]] = []
    legit_by_group: dict[str, list[list[float]]] = defaultdict(list)

    for case in selected_cases:
        legit = extract_feature_vector(
            overlay=overlay,
            prompt=case.prompt,
            anchor_text=case.anchor_text,
            layers=layers,
            max_length=args.max_length,
            device=device,
        )
        legit_row = {
            "label": "legit",
            "name": case.name,
            "anchor_group": case.anchor_group,
            "anchor_class": case.anchor_class,
            "host_case": None,
            **legit,
        }
        legit_rows.append(legit_row)
        if legit.get("status") == "ok":
            legit_by_group[case.anchor_group].append(list(legit["feature_vector"]))

        host_case = host_map[case.name]
        injected_prompt = build_injected_prompt(case, host_case)
        injected = extract_feature_vector(
            overlay=overlay,
            prompt=injected_prompt,
            anchor_text=case.anchor_text,
            layers=layers,
            max_length=args.max_length,
            device=device,
        )
        injected_rows.append(
            {
                "label": "injected",
                "name": f"{case.name}__into__{host_case.name}",
                "anchor_group": case.anchor_group,
                "anchor_class": case.anchor_class,
                "host_case": host_case.name,
                "status": injected.get("status"),
                "note": injected.get("note"),
                "span_token_count": injected.get("span_token_count"),
                "match_method": injected.get("match_method"),
                "feature_vector": injected.get("feature_vector"),
                "layer_metrics": injected.get("layer_metrics"),
                "prompt": injected_prompt,
            }
        )

    legit_vectors = [row["feature_vector"] for row in legit_rows if row.get("status") == "ok"]
    scale = compute_scale(legit_vectors)

    for row in legit_rows:
        if row.get("status") != "ok":
            row["distance_to_group_prototype"] = None
            continue
        group = str(row["anchor_group"])
        candidates = [
            vector
            for other in legit_rows
            if other.get("status") == "ok"
            and other.get("anchor_group") == group
            and other.get("name") != row.get("name")
            for vector in [other["feature_vector"]]
        ]
        if not candidates:
            row["distance_to_group_prototype"] = None
            continue
        prototype = mean_vector(candidates)
        row["distance_to_group_prototype"] = standardized_l1_distance(row["feature_vector"], prototype, scale)

    for row in injected_rows:
        if row.get("status") != "ok":
            row["distance_to_group_prototype"] = None
            continue
        group = str(row["anchor_group"])
        prototype = mean_vector(legit_by_group.get(group, []))
        if not prototype:
            row["distance_to_group_prototype"] = None
            continue
        row["distance_to_group_prototype"] = standardized_l1_distance(row["feature_vector"], prototype, scale)

    legit_scores = [
        float(row["distance_to_group_prototype"])
        for row in legit_rows
        if row.get("distance_to_group_prototype") is not None
    ]
    injected_scores = [
        float(row["distance_to_group_prototype"])
        for row in injected_rows
        if row.get("distance_to_group_prototype") is not None
    ]

    per_class_auc: dict[str, float | None] = {}
    for anchor_class in sorted({row["anchor_class"] for row in legit_rows + injected_rows}):
        class_legit = [
            float(row["distance_to_group_prototype"])
            for row in legit_rows
            if row.get("anchor_class") == anchor_class and row.get("distance_to_group_prototype") is not None
        ]
        class_injected = [
            float(row["distance_to_group_prototype"])
            for row in injected_rows
            if row.get("anchor_class") == anchor_class and row.get("distance_to_group_prototype") is not None
        ]
        per_class_auc[anchor_class] = compute_auc(class_legit, class_injected)

    result = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": args.model_name,
            "device": str(device),
            "profile": args.profile,
            "group_case_cap": args.group_case_cap,
            "max_length": args.max_length,
            "probe_layers": layers,
            "feature_metrics": list(FEATURE_METRICS),
        },
        "summary": {
            "detection_auc": compute_auc(legit_scores, injected_scores),
            "legit_mean_distance": _safe_mean(legit_scores),
            "injected_mean_distance": _safe_mean(injected_scores),
            "n_legit_ok": len(legit_scores),
            "n_injected_ok": len(injected_scores),
            "per_class_auc": per_class_auc,
        },
        "samples": legit_rows + injected_rows,
    }

    archive_dir = ROOT / "archive"
    archive_dir.mkdir(exist_ok=True)
    out_path = archive_dir / f"qwen35_4b_injection_geometry_{args.profile}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_json={out_path.relative_to(ROOT)}")
    print("===FINAL_RESULT===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
