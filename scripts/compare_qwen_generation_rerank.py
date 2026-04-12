from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_rerank_cases import QwenRerankCase, make_qwen_rerank_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.model.qwen_candidate_rerank import (
    candidate_average_logprob,
    extract_candidate_metrics,
    longest_common_prefix_length,
    reranked_candidate_score,
)


def _prefix_length(tokenizer: Any, prompt: str, full_text: str) -> int:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    return longest_common_prefix_length(prompt_ids, full_ids)


def evaluate_candidate(
    overlay: QwenAnchorOverlay,
    prompt: str,
    candidate: str,
    max_length: int,
    future_window: int,
    span_threshold: float,
    top_spans: int,
    rerank_strength: float,
) -> dict[str, Any]:
    full_text = prompt + candidate
    out, batch = overlay.analyze_texts_with_future_influence(
        [full_text],
        max_length=max_length,
        future_window=future_window,
        span_threshold=span_threshold,
        top_spans=top_spans,
    )
    prefix_length = _prefix_length(overlay.tokenizer, prompt=prompt, full_text=full_text)
    base_avg_logprob = candidate_average_logprob(
        logits=out["logits"].detach(),
        input_ids=batch["input_ids"],
        prefix_length=prefix_length,
    )
    metrics = extract_candidate_metrics(out)
    rerank_score = reranked_candidate_score(
        base_average_logprob=base_avg_logprob,
        anchor_bonus=metrics["anchor_bonus"],
        rerank_strength=rerank_strength,
    )
    return {
        "text": candidate,
        "prefix_length": prefix_length,
        "base_average_logprob": float(base_avg_logprob),
        "anchor_bonus": float(metrics["anchor_bonus"]),
        "rerank_score": float(rerank_score),
        **metrics,
    }


def evaluate_case(
    overlay: QwenAnchorOverlay,
    case: QwenRerankCase,
    max_length: int,
    future_window: int,
    span_threshold: float,
    top_spans: int,
    rerank_strength: float,
) -> dict[str, Any]:
    preferred = evaluate_candidate(
        overlay=overlay,
        prompt=case.prompt,
        candidate=case.preferred,
        max_length=max_length,
        future_window=future_window,
        span_threshold=span_threshold,
        top_spans=top_spans,
        rerank_strength=rerank_strength,
    )
    rejected = evaluate_candidate(
        overlay=overlay,
        prompt=case.prompt,
        candidate=case.rejected,
        max_length=max_length,
        future_window=future_window,
        span_threshold=span_threshold,
        top_spans=top_spans,
        rerank_strength=rerank_strength,
    )
    base_correct = preferred["base_average_logprob"] > rejected["base_average_logprob"]
    anchor_correct = preferred["rerank_score"] > rejected["rerank_score"]
    return {
        "name": case.name,
        "family": case.family,
        "expected_mode": case.expected_mode,
        "prompt": case.prompt,
        "preferred": preferred,
        "rejected": rejected,
        "base_correct": bool(base_correct),
        "anchor_correct": bool(anchor_correct),
        "base_margin": float(preferred["base_average_logprob"] - rejected["base_average_logprob"]),
        "anchor_margin": float(preferred["rerank_score"] - rejected["rerank_score"]),
        "anchor_delta_vs_base": float(
            (preferred["rerank_score"] - rejected["rerank_score"])
            - (preferred["base_average_logprob"] - rejected["base_average_logprob"])
        ),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(results),
        "base_accuracy": sum(1 for item in results if item["base_correct"]) / max(len(results), 1),
        "anchor_accuracy": sum(1 for item in results if item["anchor_correct"]) / max(len(results), 1),
        "anchor_minus_base_accuracy": (
            sum(1 for item in results if item["anchor_correct"]) - sum(1 for item in results if item["base_correct"])
        ),
        "rescued_cases": [item["name"] for item in results if (not item["base_correct"]) and item["anchor_correct"]],
        "regressed_cases": [item["name"] for item in results if item["base_correct"] and (not item["anchor_correct"])],
    }
    for mode in ("stable", "conflict"):
        subset = [item for item in results if item["expected_mode"] == mode]
        if not subset:
            continue
        summary[f"{mode}_base_accuracy"] = sum(1 for item in subset if item["base_correct"]) / len(subset)
        summary[f"{mode}_anchor_accuracy"] = sum(1 for item in subset if item["anchor_correct"]) / len(subset)
    return summary


def build_markdown_report(
    model_name: str,
    device: str,
    max_length: int,
    future_window: int,
    rerank_strength: float,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Qwen Base vs Anchor Rerank Compare",
        "",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Max length: `{max_length}`",
        f"Future window: `{future_window}`",
        f"Rerank strength: `{rerank_strength:.2f}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Base accuracy: `{summary['base_accuracy']:.4f}`",
        f"- Anchor rerank accuracy: `{summary['anchor_accuracy']:.4f}`",
        f"- Accuracy delta (anchor - base): `{summary['anchor_minus_base_accuracy']:+d}` cases",
        f"- Stable base accuracy: `{summary.get('stable_base_accuracy', 0.0):.4f}`",
        f"- Stable anchor accuracy: `{summary.get('stable_anchor_accuracy', 0.0):.4f}`",
        f"- Conflict base accuracy: `{summary.get('conflict_base_accuracy', 0.0):.4f}`",
        f"- Conflict anchor accuracy: `{summary.get('conflict_anchor_accuracy', 0.0):.4f}`",
        "",
        f"- Rescued cases: `{', '.join(summary['rescued_cases']) if summary['rescued_cases'] else 'none'}`",
        f"- Regressed cases: `{', '.join(summary['regressed_cases']) if summary['regressed_cases'] else 'none'}`",
        "",
        "## Case table",
        "",
        "| Family | Case | Mode | Base ok | Anchor ok | Base margin | Anchor margin | Δ vs base |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['family']} | {item['name']} | {item['expected_mode']} | "
            f"{'yes' if item['base_correct'] else 'no'} | {'yes' if item['anchor_correct'] else 'no'} | "
            f"{item['base_margin']:.4f} | {item['anchor_margin']:.4f} | {item['anchor_delta_vs_base']:.4f} |"
        )

    lines.extend(["", "## Candidate diagnostics", ""])
    for item in results:
        lines.append(f"### {item['name']}")
        lines.append(
            f"- preferred base logprob `{item['preferred']['base_average_logprob']:.4f}`, anchor bonus `{item['preferred']['anchor_bonus']:.4f}`, rerank `{item['preferred']['rerank_score']:.4f}`"
        )
        lines.append(
            f"- rejected base logprob `{item['rejected']['base_average_logprob']:.4f}`, anchor bonus `{item['rejected']['anchor_bonus']:.4f}`, rerank `{item['rejected']['rerank_score']:.4f}`"
        )
        lines.append(
            f"- preferred pressure `{item['preferred']['mean_contradiction_pressure']:.4f}` vs rejected `{item['rejected']['mean_contradiction_pressure']:.4f}`"
        )
        lines.append(
            f"- preferred viability `{item['preferred']['mean_viability']:.4f}` vs rejected `{item['rejected']['mean_viability']:.4f}`"
        )
        lines.append(
            f"- preferred revise gain `{item['preferred']['auxiliary_revision_revise_gain']:+d}` vs rejected `{item['rejected']['auxiliary_revision_revise_gain']:+d}`"
        )
        lines.append(
            f"- preferred continuation: `{item['preferred']['text'].strip()}`"
        )
        lines.append(
            f"- rejected continuation: `{item['rejected']['text'].strip()}`"
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Это не свободная генерация, а constrained reranking между двумя короткими продолжениями на один prompt.",
            "- Такой тест слабее настоящего decoding benchmark, но уже позволяет увидеть, даёт ли anchor-side сигнал хоть какой-то полезный приоритет поверх base model.",
            "- Если anchor rerank выигрывает хотя бы некоторые cases, следующий шаг — перенести ту же логику в мягкий logits bias или beam rerank на реальной генерации.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base Qwen scoring against anchor-assisted reranking.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--future_window", type=int, default=16)
    parser.add_argument("--span_threshold", type=float, default=0.75)
    parser.add_argument("--top_spans", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rerank_strength", type=float, default=0.35)
    parser.add_argument(
        "--case_filter",
        type=str,
        default="",
        help="Comma-separated family or case-name filters.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_generation_rerank_compare.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_generation_rerank_compare.md",
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

    filters = {item.strip() for item in args.case_filter.split(",") if item.strip()}
    cases = [
        case
        for case in make_qwen_rerank_cases()
        if not filters or case.family in filters or case.name in filters
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        result = evaluate_case(
            overlay=overlay,
            case=case,
            max_length=args.max_length,
            future_window=args.future_window,
            span_threshold=args.span_threshold,
            top_spans=args.top_spans,
            rerank_strength=args.rerank_strength,
        )
        results.append(result)
        print(
            f"{case.name}: base={'ok' if result['base_correct'] else 'miss'} "
            f"anchor={'ok' if result['anchor_correct'] else 'miss'} "
            f"base_margin={result['base_margin']:.4f} anchor_margin={result['anchor_margin']:.4f}"
        )

    summary = summarize_results(results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "future_window": args.future_window,
        "span_threshold": args.span_threshold,
        "top_spans": args.top_spans,
        "seed": args.seed,
        "rerank_strength": args.rerank_strength,
        "results": results,
        "summary": summary,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        future_window=args.future_window,
        rerank_strength=args.rerank_strength,
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
