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
    compute_cross_prompt_stability,
    compute_geometry_metrics,
    compute_mean_direction,
    extract_delta_vectors,
    match_anchor_span,
    select_representative_layers,
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
    return f"{float(value):.4f}"


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
        }

    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    layer_results: list[dict[str, Any]] = []
    for layer in layers:
        layer_hidden = hidden_states[layer][0]
        delta_vectors = extract_delta_vectors(
            hidden_states=layer_hidden,
            token_start=span_match.token_start,
            token_end=span_match.token_end,
        )
        metrics = compute_geometry_metrics(delta_vectors)
        mean_direction = compute_mean_direction(delta_vectors)
        layer_results.append(
            {
                "layer": layer,
                "hidden_state_index": layer,
                "token_count": int(span_match.token_count),
                "delta_count": int(delta_vectors.size(0)),
                "span_match": asdict(span_match),
                "metrics": {key: _to_scalar(value) for key, value in metrics.items()},
                "mean_direction": mean_direction.detach().cpu().tolist() if mean_direction is not None else None,
            }
        )

    decoded_span = overlay.tokenizer.decode(
        input_ids[span_match.token_start : span_match.token_end + 1],
        skip_special_tokens=False,
    )
    return {
        "name": case.name,
        "anchor_class": case.anchor_class,
        "anchor_group": case.anchor_group,
        "anchor_text": case.anchor_text,
        "prompt": case.prompt,
        "description": case.description,
        "status": "ok",
        "decoded_span": decoded_span,
        "token_ids": input_ids,
        "span_match": asdict(span_match),
        "layer_results": layer_results,
    }


def aggregate_results(
    results: list[dict[str, Any]],
    layers: list[int],
) -> dict[str, Any]:
    numeric_metrics = (
        "adjacent_cosine_coherence",
        "path_tortuosity",
        "rank1_explained_variance",
        "curvature_proxy",
        "mean_direction_norm",
        "mean_step_norm",
    )
    valid_results = [result for result in results if result["status"] == "ok"]
    class_layer_metrics: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    class_layer_directions: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    group_layer_directions: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    for anchor_class in sorted({result["anchor_class"] for result in valid_results}):
        class_subset = [result for result in valid_results if result["anchor_class"] == anchor_class]
        for layer in layers:
            layer_subset = []
            for result in class_subset:
                matched = next((item for item in result["layer_results"] if int(item["layer"]) == layer), None)
                if matched is not None:
                    layer_subset.append(matched)
                    if matched["mean_direction"] is not None:
                        class_layer_directions[anchor_class][layer].append(
                            torch.tensor(matched["mean_direction"], dtype=torch.float32)
                        )
                        group_layer_directions[result["anchor_group"]][layer].append(
                            torch.tensor(matched["mean_direction"], dtype=torch.float32)
                        )
            metric_summary = {
                metric_name: _summarize_numeric(
                    [entry["metrics"].get(metric_name) for entry in layer_subset]
                )
                for metric_name in numeric_metrics
            }
            token_counts = [int(entry["metrics"]["token_count"]) for entry in layer_subset]
            delta_counts = [int(entry["metrics"]["delta_count"]) for entry in layer_subset]
            class_layer_metrics[anchor_class][layer] = {
                "prompt_count": len(layer_subset),
                "token_count": _summarize_numeric([float(value) for value in token_counts]),
                "delta_count": _summarize_numeric([float(value) for value in delta_counts]),
                "computable_full_metric_count": sum(
                    1 for entry in layer_subset if entry["metrics"].get("adjacent_cosine_coherence") is not None
                ),
                "metrics": metric_summary,
                "cross_prompt_stability": compute_cross_prompt_stability(class_layer_directions[anchor_class][layer]),
            }

    group_summary: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for group_name, layer_map in group_layer_directions.items():
        for layer, directions in layer_map.items():
            group_summary[group_name][layer] = {
                "direction_count": len(directions),
                "cross_prompt_stability": compute_cross_prompt_stability(directions),
            }

    comparisons: dict[int, dict[str, float | None]] = {}
    for layer in layers:
        content = class_layer_metrics.get("content_like", {}).get(layer, {})
        procedure = class_layer_metrics.get("procedure_like", {}).get(layer, {})
        comparisons[layer] = {}
        for metric_name in ("adjacent_cosine_coherence", "path_tortuosity", "rank1_explained_variance"):
            content_mean = (((content.get("metrics") or {}).get(metric_name) or {}).get("mean"))
            procedure_mean = (((procedure.get("metrics") or {}).get(metric_name) or {}).get("mean"))
            if content_mean is None or procedure_mean is None:
                comparisons[layer][f"{metric_name}_content_minus_procedure"] = None
            else:
                comparisons[layer][f"{metric_name}_content_minus_procedure"] = float(content_mean - procedure_mean)
        content_stability = (((content.get("cross_prompt_stability") or {}).get("mean_pairwise_cosine")))
        procedure_stability = (((procedure.get("cross_prompt_stability") or {}).get("mean_pairwise_cosine")))
        if content_stability is None or procedure_stability is None:
            comparisons[layer]["cross_prompt_stability_content_minus_procedure"] = None
        else:
            comparisons[layer]["cross_prompt_stability_content_minus_procedure"] = float(
                content_stability - procedure_stability
            )

    skipped = [result for result in results if result["status"] != "ok"]
    return {
        "valid_prompt_count": len(valid_results),
        "skipped_prompt_count": len(skipped),
        "skipped_cases": skipped,
        "class_summary": class_layer_metrics,
        "group_summary": group_summary,
        "layer_comparisons": comparisons,
    }


def infer_interpretation(aggregate: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    positive_signals = 0
    evaluated_signals = 0
    details: list[str] = []
    for layer in layers:
        comparison = aggregate["layer_comparisons"][layer]
        coherence_gap = comparison.get("adjacent_cosine_coherence_content_minus_procedure")
        tortuosity_gap = comparison.get("path_tortuosity_content_minus_procedure")
        rank1_gap = comparison.get("rank1_explained_variance_content_minus_procedure")
        stability_gap = comparison.get("cross_prompt_stability_content_minus_procedure")
        layer_flags = []
        if coherence_gap is not None:
            evaluated_signals += 1
            if coherence_gap > 0.0:
                positive_signals += 1
                layer_flags.append("coherence")
        if tortuosity_gap is not None:
            evaluated_signals += 1
            if tortuosity_gap < 0.0:
                positive_signals += 1
                layer_flags.append("tortuosity")
        if rank1_gap is not None:
            evaluated_signals += 1
            if rank1_gap > 0.0:
                positive_signals += 1
                layer_flags.append("rank1")
        if stability_gap is not None:
            evaluated_signals += 1
            if stability_gap > 0.0:
                positive_signals += 1
                layer_flags.append("stability")
        details.append(
            f"Layer {layer}: supporting signals = {', '.join(layer_flags) if layer_flags else 'none'}."
        )
    if evaluated_signals == 0:
        verdict = "no_separation"
        recommendation = "refine_prompt_set"
    else:
        ratio = positive_signals / evaluated_signals
        if ratio >= 0.70:
            verdict = "clear_separation"
            recommendation = "continue"
        elif ratio >= 0.45:
            verdict = "partial_separation"
            recommendation = "refine_metric"
        else:
            verdict = "no_separation"
            recommendation = "abandon_this_direction"
    return {
        "verdict": verdict,
        "positive_signals": positive_signals,
        "evaluated_signals": evaluated_signals,
        "details": details,
        "recommended_next_step": recommendation,
    }


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    max_length: int,
    layers: list[int],
    cases: list[QwenAnchorGeometryCase],
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    interpretation: dict[str, Any],
) -> str:
    lines = [
        "# Qwen Anchor Geometry Report",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Max length: `{max_length}`",
        f"Representative layers: `{layers}`",
        "",
        "## Research question",
        "",
        "Can anchor polarity be inferred from the geometry of hidden-state assembly across the anchor span?",
        "",
        "## Design",
        "",
        f"- Prompt count: `{len(cases)}`",
        f"- Valid prompts: `{aggregate['valid_prompt_count']}`",
        f"- Skipped prompts: `{aggregate['skipped_prompt_count']}`",
        "- Geometry is computed from raw Qwen hidden states obtained through the repo overlay loader, without generation steering.",
        "- Deltas are token-to-token differences inside the matched anchor span.",
        "- `rank1_explained_variance` is implemented as the rank-1 energy fraction of the uncentered delta matrix.",
        "",
        "## Overall summary",
        "",
        f"- Verdict: `{interpretation['verdict']}`",
        f"- Positive signals: `{interpretation['positive_signals']}` / `{interpretation['evaluated_signals']}`",
        f"- Recommended next step: `{interpretation['recommended_next_step']}`",
        "",
        "## Layer-by-layer class comparison",
        "",
        "| Layer | Content coherence | Procedure coherence | Content tortuosity | Procedure tortuosity | Content rank-1 EV | Procedure rank-1 EV | Content stability | Procedure stability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        content = aggregate["class_summary"].get("content_like", {}).get(layer, {})
        procedure = aggregate["class_summary"].get("procedure_like", {}).get(layer, {})
        content_metrics = content.get("metrics", {})
        procedure_metrics = procedure.get("metrics", {})
        content_stability = (content.get("cross_prompt_stability") or {}).get("mean_pairwise_cosine")
        procedure_stability = (procedure.get("cross_prompt_stability") or {}).get("mean_pairwise_cosine")
        lines.append(
            "| {layer} | {cc} | {pc} | {ct} | {pt} | {cr} | {pr} | {cs} | {ps} |".format(
                layer=layer,
                cc=_fmt_metric(((content_metrics.get("adjacent_cosine_coherence") or {}).get("mean"))),
                pc=_fmt_metric(((procedure_metrics.get("adjacent_cosine_coherence") or {}).get("mean"))),
                ct=_fmt_metric(((content_metrics.get("path_tortuosity") or {}).get("mean"))),
                pt=_fmt_metric(((procedure_metrics.get("path_tortuosity") or {}).get("mean"))),
                cr=_fmt_metric(((content_metrics.get("rank1_explained_variance") or {}).get("mean"))),
                pr=_fmt_metric(((procedure_metrics.get("rank1_explained_variance") or {}).get("mean"))),
                cs=_fmt_metric(content_stability),
                ps=_fmt_metric(procedure_stability),
            )
        )

    lines.extend(
        [
            "",
            "## Per-prompt table",
            "",
            "| Class | Group | Case | Layer | Tokens | Deltas | Coherence | Tortuosity | Rank-1 EV | Mean dir norm |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        if result["status"] != "ok":
            continue
        for layer_result in result["layer_results"]:
            metrics = layer_result["metrics"]
            lines.append(
                "| {anchor_class} | {anchor_group} | {name} | {layer} | {token_count} | {delta_count} | {coherence} | {tortuosity} | {rank1} | {mean_norm} |".format(
                    anchor_class=result["anchor_class"],
                    anchor_group=result["anchor_group"],
                    name=result["name"],
                    layer=int(layer_result["layer"]),
                    token_count=int(metrics["token_count"]),
                    delta_count=int(metrics["delta_count"]),
                    coherence=_fmt_metric(metrics.get("adjacent_cosine_coherence")),
                    tortuosity=_fmt_metric(metrics.get("path_tortuosity")),
                    rank1=_fmt_metric(metrics.get("rank1_explained_variance")),
                    mean_norm=_fmt_metric(metrics.get("mean_direction_norm")),
                )
            )

    lines.extend(["", "## Cross-prompt stability by anchor group", ""])
    lines.extend(
        [
            "| Group | Layer | Pair count | Mean pairwise cosine |",
            "|---|---:|---:|---:|",
        ]
    )
    for group_name, layer_map in sorted(aggregate["group_summary"].items()):
        for layer in layers:
            stability = (layer_map.get(layer) or {}).get("cross_prompt_stability") or {}
            lines.append(
                "| {group} | {layer} | {pair_count} | {cosine} |".format(
                    group=group_name,
                    layer=layer,
                    pair_count=int(stability.get("pair_count") or 0),
                    cosine=_fmt_metric(stability.get("mean_pairwise_cosine")),
                )
            )

    lines.extend(["", "## Interpretation", ""])
    for detail in interpretation["details"]:
        lines.append(f"- {detail}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(
        [
            "- The prompt set is intentionally small and local, so class-level separation can still be confounded by phrase identity.",
            "- Content-like anchors remain semantically heterogeneous even after phrase grouping; phrase-level stability is therefore reported separately.",
            "- Short spans would disable curvature-style metrics, although this prompt set was chosen to keep spans at four tokens in most cases.",
            "- This experiment measures static hidden-state assembly during prompt processing only; it does not test runtime steering quality directly.",
        ]
    )
    if aggregate["skipped_cases"]:
        lines.extend(["", "## Skipped cases", ""])
        for skipped in aggregate["skipped_cases"]:
            lines.append(f"- `{skipped['name']}`: `{skipped['skip_reason']}`")
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    output_json: Path,
    output_md: Path,
    payload: dict[str, Any],
    report_text: str,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(report_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen anchor-span hidden-state geometry.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_anchor_geometry_probe.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_geometry_report.md",
    )
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device(args.device)
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model,
        cfg=TOY_CONFIG,
        device=device,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
    )
    overlay.eval()

    num_hidden_layers = int(getattr(overlay.base_model.config, "num_hidden_layers"))
    layers = select_representative_layers(num_hidden_layers=num_hidden_layers, count=4)
    cases = make_qwen_anchor_geometry_cases()
    results = [
        analyze_case_geometry(
            overlay=overlay,
            case=case,
            layers=layers,
            max_length=args.max_length,
            device=device,
        )
        for case in cases
    ]
    aggregate = aggregate_results(results=results, layers=layers)
    interpretation = infer_interpretation(aggregate=aggregate, layers=layers)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "num_hidden_layers": num_hidden_layers,
        "layers": layers,
        "cases": [asdict(case) for case in cases],
        "results": results,
        "aggregate": aggregate,
        "interpretation": interpretation,
    }
    report_text = build_markdown_report(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        layers=layers,
        cases=cases,
        results=results,
        aggregate=aggregate,
        interpretation=interpretation,
    )
    write_outputs(
        output_json=args.output_json,
        output_md=args.output_md,
        payload=payload,
        report_text=report_text,
    )
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
