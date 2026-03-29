from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_probe_cases import make_qwen_probe_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.model.future_span_hints import (
    build_future_hint_candidates,
    compute_span_anchor_overlap,
    extract_high_influence_spans,
    safe_decode_token,
)


def collect_case_result(
    overlay: QwenAnchorOverlay,
    case_name: str,
    case_family: str,
    case_description: str,
    case_prompt: str,
    expected_mode: str,
    max_length: int,
    future_window: int,
    top_k: int,
    span_threshold: float,
    top_spans: int,
) -> dict[str, Any]:
    out, batch = overlay.analyze_texts_with_future_influence(
        [case_prompt],
        max_length=max_length,
        future_window=future_window,
    )
    diag = out["anchor_diagnostics"]
    influence = out["future_influence"]
    scores = influence["scores"][0]
    input_ids = batch["input_ids"][0]
    valid_len = int(batch["attention_mask"][0].sum().item()) if "attention_mask" in batch else int(input_ids.numel())
    trimmed_scores = scores[:valid_len]
    trimmed_ids = input_ids[:valid_len]

    k = min(top_k, valid_len)
    top_values, top_indices = torch.topk(trimmed_scores, k=k)
    top_tokens = [
        {
            "position": int(pos.item()),
            "token_id": int(trimmed_ids[pos].item()),
            "token_text": safe_decode_token(overlay.tokenizer, int(trimmed_ids[pos].item())),
            "score": float(val.item()),
        }
        for val, pos in zip(top_values, top_indices)
    ]

    active_anchor_spans = [
        {
            "start": max(0, min(int(anchor.start_idx), valid_len - 1)),
            "end": max(0, min(int(anchor.end_idx), valid_len - 1)),
        }
        for anchor in out["active_anchors"][0]
        if valid_len > 0
    ]
    anchor_positions = sorted({span["end"] for span in active_anchor_spans})
    anchor_scores = [float(trimmed_scores[pos].item()) for pos in anchor_positions]
    future_spans = extract_high_influence_spans(
        scores=trimmed_scores,
        input_ids=trimmed_ids,
        tokenizer=overlay.tokenizer,
        min_score=span_threshold,
        top_spans=top_spans,
    )
    future_hint_candidates = build_future_hint_candidates(future_spans, active_anchor_spans)
    overlap = compute_span_anchor_overlap(future_spans, active_anchor_spans)
    return {
        "name": case_name,
        "family": case_family,
        "description": case_description,
        "expected_mode": expected_mode,
        "tokens": valid_len,
        "num_active": int(diag["num_active"]),
        "mean_contradiction_pressure": float(diag["mean_contradiction_pressure"]),
        "mean_viability": float(diag["mean_viability"]),
        "future_loss": float(influence["loss"]),
        "future_window": int(influence["target_window"]),
        "mean_future_influence": float(trimmed_scores.mean().item()),
        "max_future_influence": float(trimmed_scores.max().item()),
        "anchor_position_mean_future_influence": (
            sum(anchor_scores) / len(anchor_scores) if anchor_scores else 0.0
        ),
        "anchor_positions": anchor_positions,
        "active_anchor_spans": active_anchor_spans,
        "future_spans": future_spans,
        "future_hint_candidates": future_hint_candidates,
        **overlap,
        "top_future_tokens": top_tokens,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(results),
        "stable_count": sum(1 for item in results if item["expected_mode"] == "stable"),
        "conflict_count": sum(1 for item in results if item["expected_mode"] == "conflict"),
    }
    for mode in ("stable", "conflict"):
        subset = [item for item in results if item["expected_mode"] == mode]
        if not subset:
            continue
        summary[f"{mode}_mean_future_influence"] = sum(item["mean_future_influence"] for item in subset) / len(subset)
        summary[f"{mode}_mean_anchor_future_influence"] = (
            sum(item["anchor_position_mean_future_influence"] for item in subset) / len(subset)
        )
        summary[f"{mode}_mean_future_span_overlap"] = (
            sum(item.get("future_span_overlap_ratio", 0.0) for item in subset) / len(subset)
        )
        summary[f"{mode}_mean_anchor_span_overlap"] = (
            sum(item.get("anchor_span_overlap_ratio", 0.0) for item in subset) / len(subset)
        )
        summary[f"{mode}_mean_future_loss"] = sum(item["future_loss"] for item in subset) / len(subset)
    if "stable_mean_future_influence" in summary and "conflict_mean_future_influence" in summary:
        summary["future_influence_gap_conflict_minus_stable"] = (
            summary["conflict_mean_future_influence"] - summary["stable_mean_future_influence"]
        )
    if "stable_mean_anchor_future_influence" in summary and "conflict_mean_anchor_future_influence" in summary:
        summary["anchor_future_influence_gap_conflict_minus_stable"] = (
            summary["conflict_mean_anchor_future_influence"] - summary["stable_mean_anchor_future_influence"]
        )
    if "stable_mean_future_span_overlap" in summary and "conflict_mean_future_span_overlap" in summary:
        summary["future_span_overlap_gap_conflict_minus_stable"] = (
            summary["conflict_mean_future_span_overlap"] - summary["stable_mean_future_span_overlap"]
        )
    return summary


def build_markdown_report(
    model_name: str,
    device: str,
    max_length: int,
    future_window: int,
    span_threshold: float,
    top_spans: int,
    seed: int,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Qwen Future Influence Probe",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Max length: `{max_length}`",
        f"Future window: `{future_window}`",
        f"Span threshold: `{span_threshold:.2f}`",
        f"Top spans per case: `{top_spans}`",
        f"Seed: `{seed}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Stable cases: `{summary['stable_count']}`",
        f"- Conflict cases: `{summary['conflict_count']}`",
    ]
    if "future_influence_gap_conflict_minus_stable" in summary:
        lines.append(
            f"- Conflict minus stable mean future influence gap: `{summary['future_influence_gap_conflict_minus_stable']:.4f}`"
        )
    if "anchor_future_influence_gap_conflict_minus_stable" in summary:
        lines.append(
            f"- Conflict minus stable active-anchor future influence gap: `{summary['anchor_future_influence_gap_conflict_minus_stable']:.4f}`"
        )
    if "future_span_overlap_gap_conflict_minus_stable" in summary:
        lines.append(
            f"- Conflict minus stable future-span overlap gap: `{summary['future_span_overlap_gap_conflict_minus_stable']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Case table",
            "",
            "| Family | Case | Expected | Tokens | Active | Mean future influence | Anchor-position mean | Span overlap | Max influence | Future loss |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        lines.append(
            "| {family} | {name} | {expected_mode} | {tokens} | {num_active} | {mean_future_influence:.4f} | "
            "{anchor_position_mean_future_influence:.4f} | {future_span_overlap_ratio:.4f} | {max_future_influence:.4f} | {future_loss:.4f} |".format(**item)
        )

    lines.extend(["", "## Top future-influence tokens", ""])
    for item in results:
        lines.append(f"### {item['name']}")
        for token in item["top_future_tokens"]:
            lines.append(
                f"- pos `{token['position']}` | token `{token['token_text']}` | id `{token['token_id']}` | score `{token['score']:.4f}`"
            )
        lines.append("")

    lines.extend(["## High future-influence spans", ""])
    for item in results:
        lines.append(f"### {item['name']}")
        if not item["future_spans"]:
            lines.append("- no spans crossed the configured threshold")
        for span in item["future_spans"]:
            lines.append(
                f"- span `{span['start']}-{span['end']}` | mean `{span['mean_score']:.4f}` | max `{span['max_score']:.4f}` | text `{span['text']}`"
            )
        if item["active_anchor_spans"]:
            anchor_text = ", ".join(
                f"{span['start']}-{span['end']}" for span in item["active_anchor_spans"]
            )
            lines.append(f"- active anchor spans: `{anchor_text}`")
        else:
            lines.append("- active anchor spans: none")
        lines.append(
            f"- future-span overlap ratio: `{item['future_span_overlap_ratio']:.4f}` | anchor-span overlap ratio: `{item['anchor_span_overlap_ratio']:.4f}`"
        )
        if item["future_hint_candidates"]:
            lines.append("- proposal-like future hint spans:")
            for span in item["future_hint_candidates"]:
                lines.append(
                    f"  - `{span['start']}-{span['end']}` | mean `{span['mean_score']:.4f}` | text `{span['text']}`"
                )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This report is an experimental midpoint between delta-hidden heuristics and full leave-one-out KL.",
            "- Scores are based on gradient influence of token positions on a future autoregressive loss window.",
            "- High-scoring positions are candidates for semantically important context even when local hidden-state jumps are ambiguous.",
            "- Grouped high-influence spans help test whether future-attribution concentrates on the same regions as current active anchors or highlights missed context spans.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run future-gradient influence diagnostics on top of Qwen hidden states.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--future_window", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--span_threshold", type=float, default=0.75)
    parser.add_argument("--top_spans", type=int, default=5)
    parser.add_argument("--case_filter", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_future_influence_probe.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_future_influence_probe.md",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
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
    )
    overlay.eval()

    print("=== Qwen Future Influence Probe ===")
    print(f"model={args.model}")
    print(f"device={args.device}")
    print()

    results: list[dict[str, Any]] = []
    case_filters = [part.strip() for part in args.case_filter.split(",") if part.strip()]
    cases = make_qwen_probe_cases()
    if case_filters:
        cases = [
            case
            for case in cases
            if any(
                needle.lower() in case.name.lower() or needle.lower() in case.family.lower()
                for needle in case_filters
            )
        ]
    for case in cases:
        result = collect_case_result(
            overlay=overlay,
            case_name=case.name,
            case_family=case.family,
            case_description=case.description,
            case_prompt=case.prompt,
            expected_mode=case.expected_mode,
            max_length=args.max_length,
            future_window=args.future_window,
            top_k=args.top_k,
            span_threshold=args.span_threshold,
            top_spans=args.top_spans,
        )
        results.append(result)
        print(f"--- {case.name} ---")
        print(f"family={case.family}")
        print(f"expected_mode={case.expected_mode}")
        print(f"mean_future_influence={result['mean_future_influence']:.4f}")
        print(f"anchor_position_mean_future_influence={result['anchor_position_mean_future_influence']:.4f}")
        print(f"future_span_overlap_ratio={result['future_span_overlap_ratio']:.4f}")
        print(f"max_future_influence={result['max_future_influence']:.4f}")
        print(f"future_loss={result['future_loss']:.4f}")
        print()

    summary = summarize_results(results)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "future_window": args.future_window,
        "seed": args.seed,
        "results": results,
        "summary": summary,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        future_window=args.future_window,
        span_threshold=args.span_threshold,
        top_spans=args.top_spans,
        seed=args.seed,
        results=results,
        summary=summary,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
