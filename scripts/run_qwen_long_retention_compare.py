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

from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay


DEFAULT_PROMPT = (
    "You are a vegan chef. Write a detailed weekly meal plan with recipes for each day."
)
DEFAULT_POSITIVE_KEYWORDS = [
    "vegan",
    "plant-based",
    "tofu",
    "lentil",
    "lentils",
    "chickpea",
    "chickpeas",
    "bean",
    "beans",
    "vegetable",
    "vegetables",
    "mushroom",
    "mushrooms",
    "dairy-free",
]
DEFAULT_NEGATIVE_KEYWORDS = [
    "meat",
    "chicken",
    "beef",
    "pork",
    "bacon",
    "fish",
    "salmon",
    "tuna",
    "shrimp",
    "egg",
    "eggs",
    "cheese",
    "butter",
    "milk",
    "cream",
    "yogurt",
    "sausage",
    "ham",
]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    continuation_ids = generated[0, input_ids.size(1) :]
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
    positive_hits = {token: lowered.count(token) for token in positive_keywords if token in lowered}
    negative_hits = {token: lowered.count(token) for token in negative_keywords if token in lowered}
    first_negative = None
    for token in negative_keywords:
        idx = lowered.find(token)
        if idx >= 0 and (first_negative is None or idx < first_negative["char_index"]):
            first_negative = {"token": token, "char_index": idx}
    first_positive = None
    for token in positive_keywords:
        idx = lowered.find(token)
        if idx >= 0 and (first_positive is None or idx < first_positive["char_index"]):
            first_positive = {"token": token, "char_index": idx}
    lexical_score = float(sum(positive_hits.values()) - sum(negative_hits.values()))
    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "positive_total": int(sum(positive_hits.values())),
        "negative_total": int(sum(negative_hits.values())),
        "first_positive": first_positive,
        "first_negative": first_negative,
        "lexical_score": lexical_score,
    }


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    prompt: str,
    max_new_tokens: int,
    max_length: int,
    conflict_threshold: float,
    bias_scale: float,
    base: dict[str, Any],
    anchor: dict[str, Any],
    base_analysis: dict[str, Any],
    anchor_analysis: dict[str, Any],
) -> str:
    active_bias_steps = sum(
        1 for step in anchor["steps"] if step.get("bias_nonzero_anchors", 0) > 0
    )
    lines = [
        "# Qwen Long Retention Compare",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Max new tokens: `{max_new_tokens}`",
        f"Max length: `{max_length}`",
        f"Conflict threshold: `{conflict_threshold:.2f}`",
        f"Bias scale: `{bias_scale:.2f}`",
        "",
        "## Prompt",
        "",
        f"> {prompt}",
        "",
        "## Summary",
        "",
        f"- Base lexical score: `{base_analysis['lexical_score']:.2f}`",
        f"- Anchor lexical score: `{anchor_analysis['lexical_score']:.2f}`",
        f"- Base positive hits: `{base_analysis['positive_total']}`",
        f"- Anchor positive hits: `{anchor_analysis['positive_total']}`",
        f"- Base negative hits: `{base_analysis['negative_total']}`",
        f"- Anchor negative hits: `{anchor_analysis['negative_total']}`",
        f"- Anchor bias active steps: `{active_bias_steps}`",
        f"- Continuations identical: `{'yes' if base['continuation_text'] == anchor['continuation_text'] else 'no'}`",
        "",
        "## First keyword events",
        "",
        f"- Base first positive: `{base_analysis['first_positive']}`",
        f"- Base first negative: `{base_analysis['first_negative']}`",
        f"- Anchor first positive: `{anchor_analysis['first_positive']}`",
        f"- Anchor first negative: `{anchor_analysis['first_negative']}`",
        "",
        "## Base continuation",
        "",
        base["continuation_text"].strip() or "_empty_",
        "",
        "## Anchor-biased continuation",
        "",
        anchor["continuation_text"].strip() or "_empty_",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a long base vs anchor-biased Qwen compare.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conflict_threshold", type=float, default=0.55)
    parser.add_argument("--bias_scale", type=float, default=1.50)
    parser.add_argument(
        "--positive_keywords",
        type=str,
        default=",".join(DEFAULT_POSITIVE_KEYWORDS),
    )
    parser.add_argument(
        "--negative_keywords",
        type=str,
        default=",".join(DEFAULT_NEGATIVE_KEYWORDS),
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_long_retention_compare.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_long_retention_compare.md",
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

    positive_keywords = _split_csv(args.positive_keywords)
    negative_keywords = _split_csv(args.negative_keywords)

    base = generate_base(
        overlay=overlay,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    anchor = overlay.generate_with_anchor_bias(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        conflict_threshold=args.conflict_threshold,
        bias_scale=args.bias_scale,
        greedy=True,
    )
    base_analysis = analyze_keywords(
        base["continuation_text"],
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
    )
    anchor_analysis = analyze_keywords(
        anchor["continuation_text"],
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
    )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "prompt": args.prompt,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "conflict_threshold": args.conflict_threshold,
        "bias_scale": args.bias_scale,
        "seed": args.seed,
        "positive_keywords": positive_keywords,
        "negative_keywords": negative_keywords,
        "base": base,
        "anchor": anchor,
        "base_analysis": base_analysis,
        "anchor_analysis": anchor_analysis,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        conflict_threshold=args.conflict_threshold,
        bias_scale=args.bias_scale,
        base=base,
        anchor=anchor,
        base_analysis=base_analysis,
        anchor_analysis=anchor_analysis,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(report, encoding="utf-8")

    print(f"base_lexical_score={base_analysis['lexical_score']:.2f}")
    print(f"anchor_lexical_score={anchor_analysis['lexical_score']:.2f}")
    print(
        "anchor_bias_active_steps="
        f"{sum(1 for step in anchor['steps'] if step.get('bias_nonzero_anchors', 0) > 0)}"
    )
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
