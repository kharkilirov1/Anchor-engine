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

from src.data.qwen_probe_cases import QwenProbeCase, make_qwen_probe_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay


def lexical_consistency_score(family: str, text: str) -> float:
    lowered = text.lower()
    if family == "api_framework":
        positive = ["fastapi", "async", "pydantic"]
        negative = ["django", "synchronous", "template"]
    elif family == "quantifier":
        positive = ["for all", "universal", "every"]
        negative = ["witness", "there exists", "existential"]
    else:
        positive = []
        negative = []
    score = 0.0
    for token in positive:
        if token in lowered:
            score += 1.0
    for token in negative:
        if token in lowered:
            score -= 1.0
    return score


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
    encoded = tokenizer([prompt], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generated = input_ids
    generated_mask = attention_mask
    steps: list[dict[str, Any]] = []
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
        steps.append(
            {
                "token_id": int(next_token.item()),
                "token_text": tokenizer.decode([int(next_token.item())], skip_special_tokens=False),
            }
        )
        if int(next_token.item()) == int(getattr(tokenizer, "eos_token_id", -1)):
            break
        if generated.size(1) >= max_length:
            break
    continuation_ids = generated[0, input_ids.size(1) :]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "continuation_text": continuation_text,
        "steps": steps,
    }


def evaluate_case(
    overlay: QwenAnchorOverlay,
    case: QwenProbeCase,
    max_new_tokens: int,
    max_length: int,
    conflict_threshold: float,
    bias_scale: float,
) -> dict[str, Any]:
    base = generate_base(
        overlay=overlay,
        prompt=case.prompt,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
    )
    anchor = overlay.generate_with_anchor_bias(
        prompt=case.prompt,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        conflict_threshold=conflict_threshold,
        bias_scale=bias_scale,
        greedy=True,
    )
    base_score = lexical_consistency_score(case.family, base["continuation_text"])
    anchor_score = lexical_consistency_score(case.family, anchor["continuation_text"])
    return {
        "name": case.name,
        "family": case.family,
        "expected_mode": case.expected_mode,
        "base": base,
        "anchor": anchor,
        "base_score": float(base_score),
        "anchor_score": float(anchor_score),
        "anchor_minus_base_score": float(anchor_score - base_score),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(results),
        "base_mean_score": sum(item["base_score"] for item in results) / max(len(results), 1),
        "anchor_mean_score": sum(item["anchor_score"] for item in results) / max(len(results), 1),
        "anchor_minus_base_mean": (
            sum(item["anchor_minus_base_score"] for item in results) / max(len(results), 1)
        ),
        "anchor_better_cases": [item["name"] for item in results if item["anchor_score"] > item["base_score"]],
        "anchor_worse_cases": [item["name"] for item in results if item["anchor_score"] < item["base_score"]],
    }


def build_markdown_report(
    model_name: str,
    device: str,
    max_new_tokens: int,
    conflict_threshold: float,
    bias_scale: float,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Qwen Anchor-Biased Generation Compare",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Max new tokens: `{max_new_tokens}`",
        f"Conflict threshold: `{conflict_threshold:.2f}`",
        f"Bias scale: `{bias_scale:.2f}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Base mean lexical consistency score: `{summary['base_mean_score']:.4f}`",
        f"- Anchor mean lexical consistency score: `{summary['anchor_mean_score']:.4f}`",
        f"- Anchor minus base mean score: `{summary['anchor_minus_base_mean']:.4f}`",
        f"- Anchor better cases: `{', '.join(summary['anchor_better_cases']) if summary['anchor_better_cases'] else 'none'}`",
        f"- Anchor worse cases: `{', '.join(summary['anchor_worse_cases']) if summary['anchor_worse_cases'] else 'none'}`",
        "",
        "## Case table",
        "",
        "| Family | Case | Mode | Base score | Anchor score | Δ |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['family']} | {item['name']} | {item['expected_mode']} | "
            f"{item['base_score']:.2f} | {item['anchor_score']:.2f} | {item['anchor_minus_base_score']:+.2f} |"
        )
    lines.extend(["", "## Generated continuations", ""])
    for item in results:
        lines.append(f"### {item['name']}")
        lines.append(f"- base: `{item['base']['continuation_text'].strip()}`")
        lines.append(f"- anchor: `{item['anchor']['continuation_text'].strip()}`")
        lines.append(
            f"- anchor bias active steps: `{sum(1 for step in item['anchor']['steps'] if step.get('bias_nonzero_anchors', 0) > 0)}`"
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Это уже не rerank двух вручную заданных продолжений, а реальная greedy generation с token-level anchor bias.",
            "- Метрика здесь грубая и лексическая, поэтому её нельзя считать финальным доказательством. Но она позволяет быстро проверить, толкает ли вмешательство модель в ожидаемую semantic сторону.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base greedy generation vs anchor-biased generation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conflict_threshold", type=float, default=0.55)
    parser.add_argument("--bias_scale", type=float, default=1.50)
    parser.add_argument("--case_filter", type=str, default="api_framework,quantifier")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_anchor_generation_compare.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_generation_compare.md",
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
    cases = [case for case in make_qwen_probe_cases() if not filters or case.family in filters or case.name in filters]
    results: list[dict[str, Any]] = []
    for case in cases:
        result = evaluate_case(
            overlay=overlay,
            case=case,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            conflict_threshold=args.conflict_threshold,
            bias_scale=args.bias_scale,
        )
        results.append(result)
        print(
            f"{case.name}: base_score={result['base_score']:.2f} "
            f"anchor_score={result['anchor_score']:.2f} delta={result['anchor_minus_base_score']:+.2f}"
        )

    summary = summarize_results(results)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "conflict_threshold": args.conflict_threshold,
        "bias_scale": args.bias_scale,
        "seed": args.seed,
        "results": results,
        "summary": summary,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        conflict_threshold=args.conflict_threshold,
        bias_scale=args.bias_scale,
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
