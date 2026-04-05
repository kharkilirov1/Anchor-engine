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
    branch_aware_candidate_score,
    candidate_average_logprob,
    extract_candidate_metrics,
    extract_tree_candidate_metrics,
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
    tree_strength: float,
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
    anchor_metrics = extract_candidate_metrics(out)
    tree_metrics = extract_tree_candidate_metrics(out)
    anchor_score = reranked_candidate_score(
        base_average_logprob=base_avg_logprob,
        anchor_bonus=anchor_metrics["anchor_bonus"],
        rerank_strength=rerank_strength,
    )
    branch_score = branch_aware_candidate_score(
        base_average_logprob=base_avg_logprob,
        anchor_bonus=anchor_metrics["anchor_bonus"],
        tree_bonus=tree_metrics["tree_bonus"],
        rerank_strength=rerank_strength,
        tree_strength=tree_strength,
    )
    return {
        "text": candidate,
        "prefix_length": prefix_length,
        "base_average_logprob": float(base_avg_logprob),
        "anchor_rerank_score": float(anchor_score),
        "branch_score": float(branch_score),
        **anchor_metrics,
        **tree_metrics,
    }


def evaluate_case(
    overlay: QwenAnchorOverlay,
    case: QwenRerankCase,
    max_length: int,
    future_window: int,
    span_threshold: float,
    top_spans: int,
    rerank_strength: float,
    tree_strength: float,
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
        tree_strength=tree_strength,
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
        tree_strength=tree_strength,
    )
    return {
        "name": case.name,
        "family": case.family,
        "expected_mode": case.expected_mode,
        "prompt": case.prompt,
        "preferred": preferred,
        "rejected": rejected,
        "base_correct": preferred["base_average_logprob"] > rejected["base_average_logprob"],
        "anchor_correct": preferred["anchor_rerank_score"] > rejected["anchor_rerank_score"],
        "branch_correct": preferred["branch_score"] > rejected["branch_score"],
        "base_margin": float(preferred["base_average_logprob"] - rejected["base_average_logprob"]),
        "anchor_margin": float(preferred["anchor_rerank_score"] - rejected["anchor_rerank_score"]),
        "branch_margin": float(preferred["branch_score"] - rejected["branch_score"]),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(results),
        "base_accuracy": sum(1 for item in results if item["base_correct"]) / max(len(results), 1),
        "anchor_accuracy": sum(1 for item in results if item["anchor_correct"]) / max(len(results), 1),
        "branch_accuracy": sum(1 for item in results if item["branch_correct"]) / max(len(results), 1),
        "branch_minus_base_accuracy": (
            sum(1 for item in results if item["branch_correct"]) - sum(1 for item in results if item["base_correct"])
        ),
        "branch_minus_anchor_accuracy": (
            sum(1 for item in results if item["branch_correct"]) - sum(1 for item in results if item["anchor_correct"])
        ),
        "branch_rescued_cases": [
            item["name"] for item in results if (not item["anchor_correct"]) and item["branch_correct"]
        ],
        "branch_regressed_cases": [
            item["name"] for item in results if item["anchor_correct"] and (not item["branch_correct"])
        ],
    }


def build_markdown_report(
    model_name: str,
    device: str,
    max_length: int,
    future_window: int,
    rerank_strength: float,
    tree_strength: float,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Qwen Branch-Aware Tree Rerank Compare",
        "",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Max length: `{max_length}`",
        f"Future window: `{future_window}`",
        f"Anchor rerank strength: `{rerank_strength:.2f}`",
        f"Tree rerank strength: `{tree_strength:.2f}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Base accuracy: `{summary['base_accuracy']:.4f}`",
        f"- Anchor rerank accuracy: `{summary['anchor_accuracy']:.4f}`",
        f"- Branch-aware accuracy: `{summary['branch_accuracy']:.4f}`",
        f"- Branch minus base: `{summary['branch_minus_base_accuracy']:+d}` cases",
        f"- Branch minus anchor: `{summary['branch_minus_anchor_accuracy']:+d}` cases",
        f"- Branch rescued cases: `{', '.join(summary['branch_rescued_cases']) if summary['branch_rescued_cases'] else 'none'}`",
        f"- Branch regressed cases: `{', '.join(summary['branch_regressed_cases']) if summary['branch_regressed_cases'] else 'none'}`",
        "",
        "## Case table",
        "",
        "| Family | Case | Mode | Base ok | Anchor ok | Branch ok | Base margin | Anchor margin | Branch margin |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['family']} | {item['name']} | {item['expected_mode']} | "
            f"{'yes' if item['base_correct'] else 'no'} | {'yes' if item['anchor_correct'] else 'no'} | {'yes' if item['branch_correct'] else 'no'} | "
            f"{item['base_margin']:.4f} | {item['anchor_margin']:.4f} | {item['branch_margin']:.4f} |"
        )

    lines.extend(["", "## Candidate diagnostics", ""])
    for item in results:
        lines.append(f"### {item['name']}")
        lines.append(
            f"- preferred base `{item['preferred']['base_average_logprob']:.4f}`, anchor `{item['preferred']['anchor_rerank_score']:.4f}`, branch `{item['preferred']['branch_score']:.4f}`"
        )
        lines.append(
            f"- rejected base `{item['rejected']['base_average_logprob']:.4f}`, anchor `{item['rejected']['anchor_rerank_score']:.4f}`, branch `{item['rejected']['branch_score']:.4f}`"
        )
        lines.append(
            f"- preferred tree bonus `{item['preferred']['tree_bonus']:.4f}` vs rejected `{item['rejected']['tree_bonus']:.4f}`"
        )
        lines.append(
            f"- preferred coverage `{item['preferred']['tree_coverage']:.4f}` vs rejected `{item['rejected']['tree_coverage']:.4f}`"
        )
        lines.append(
            f"- preferred drift `{item['preferred']['tree_drift_score']:.4f}` vs rejected `{item['rejected']['tree_drift_score']:.4f}`"
        )
        lines.append(
            f"- preferred best repair `{item['preferred']['tree_best_repair_gain']:.4f}` vs rejected `{item['rejected']['tree_best_repair_gain']:.4f}`"
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This prototype still uses short candidate continuations, but now scores them with tree-level structure rather than only local anchor signals.",
            "- A branch wins when it keeps better expected-tree coverage, lower drift, and stronger repair utility on top of the base and anchor rerank signals.",
            "- If this branch-aware scorer shows cleaner wins than anchor-only reranking, the next step is to move from offline rerank to online branch selection during generation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base, anchor rerank, and branch-aware tree rerank on Qwen candidate cases.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--future_window", type=int, default=16)
    parser.add_argument("--span_threshold", type=float, default=0.75)
    parser.add_argument("--top_spans", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rerank_strength", type=float, default=0.35)
    parser.add_argument("--tree_strength", type=float, default=0.45)
    parser.add_argument("--case_filter", type=str, default="", help="Comma-separated family or case-name filters.")
    parser.add_argument("--output_json", type=Path, default=ROOT / "archive" / "qwen_tree_branch_rerank_compare.json")
    parser.add_argument("--output_md", type=Path, default=ROOT / "docs" / "research" / "qwen_tree_branch_rerank_compare.md")
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
    cases = [case for case in make_qwen_rerank_cases() if not filters or case.family in filters or case.name in filters]

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
            tree_strength=args.tree_strength,
        )
        results.append(result)
        print(
            f"{case.name}: base={'ok' if result['base_correct'] else 'miss'} "
            f"anchor={'ok' if result['anchor_correct'] else 'miss'} "
            f"branch={'ok' if result['branch_correct'] else 'miss'} "
            f"branch_margin={result['branch_margin']:.4f}"
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
        "tree_strength": args.tree_strength,
        "results": results,
        "summary": summary,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        future_window=args.future_window,
        rerank_strength=args.rerank_strength,
        tree_strength=args.tree_strength,
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

