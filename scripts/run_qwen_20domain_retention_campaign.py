"""Run 20-domain retention campaign: BASE vs ANCHOR for each domain.

Loads model once, runs all domains sequentially, saves per-domain JSON
and a summary table.  Designed for Qwen3.5-4B on a single GPU.

Usage:
    python scripts/run_qwen_20domain_retention_campaign.py --model Qwen/Qwen2.5-3B-Instruct
    python scripts/run_qwen_20domain_retention_campaign.py --device cpu --max_new_tokens 100
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.retention_domains import RETENTION_DOMAINS, RetentionDomain
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.model.qwen_generation_bias import (
    BiasDomainProfile,
    build_bias_token_weights,
    get_profile_by_name,
)


# ── Scoring ─────────────────────────────────────────────────────────

def analyze_keywords(
    text: str,
    positive_keywords: tuple[str, ...] | list[str],
    negative_keywords: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    lowered = text.lower()

    positive_hits: dict[str, int] = {}
    for kw in positive_keywords:
        count = lowered.count(kw.lower())
        if count > 0:
            positive_hits[kw] = count

    negative_hits: dict[str, int] = {}
    for kw in negative_keywords:
        count = lowered.count(kw.lower())
        if count > 0:
            negative_hits[kw] = count

    lexical_score = float(sum(positive_hits.values()) - sum(negative_hits.values()))

    # Degeneracy detection
    word_tokens = re.findall(r"[a-zA-Z_]+", lowered)
    bigrams = list(zip(word_tokens, word_tokens[1:]))
    repeated_bigram_ratio = 0.0
    if bigrams:
        repeated_bigram_ratio = 1.0 - (len(set(bigrams)) / max(len(bigrams), 1))

    sentences = [s.strip() for s in re.split(r"[.!?\n]+", lowered) if s.strip()]
    sentence_counts: dict[str, int] = {}
    max_sentence_repeat = 0
    for s in sentences:
        sentence_counts[s] = sentence_counts.get(s, 0) + 1
        max_sentence_repeat = max(max_sentence_repeat, sentence_counts[s])

    degeneracy_penalty = float(
        8.0 * repeated_bigram_ratio + 1.5 * max(0, max_sentence_repeat - 1)
    )
    quality_score = float(
        lexical_score - degeneracy_penalty - 1.5 * sum(negative_hits.values())
    )

    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "positive_total": int(sum(positive_hits.values())),
        "negative_total": int(sum(negative_hits.values())),
        "lexical_score": lexical_score,
        "repeated_bigram_ratio": float(repeated_bigram_ratio),
        "max_sentence_repeat": int(max_sentence_repeat),
        "degeneracy_penalty": float(degeneracy_penalty),
        "quality_score": quality_score,
    }


# ── Base generation (greedy, no anchor) ─────────────────────────────

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
        [prompt], padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
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
        steps.append({
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
        })
        if token_id == int(getattr(tokenizer, "eos_token_id", -1)):
            break
        if generated.size(1) >= max_length:
            break

    continuation_ids = generated[0, input_ids.size(1):]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "continuation_text": continuation_text,
        "num_tokens": len(steps),
    }


# ── Single domain run ──────────────────────────────────────────────

def run_single_domain(
    overlay: QwenAnchorOverlay,
    domain: RetentionDomain,
    *,
    max_new_tokens: int,
    max_length: int,
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
) -> dict[str, Any]:
    effective_tokens = domain.max_new_tokens if domain.max_new_tokens else max_new_tokens

    print(f"\n{'='*60}")
    print(f"  Domain: {domain.name}")
    print(f"  Profile: {domain.bias_profile_name}")
    print(f"  Tokens: {effective_tokens}")
    print(f"{'='*60}")

    # ── BASE ──
    t0 = time.time()
    print("  [BASE] generating...")
    base = generate_base(
        overlay, domain.prompt,
        max_new_tokens=effective_tokens, max_length=max_length,
    )
    base_time = time.time() - t0
    print(f"  [BASE] done in {base_time:.1f}s, {base['num_tokens']} tokens")

    # ── ANCHOR ──
    t0 = time.time()
    print("  [ANCHOR] generating...")
    anchor = overlay.generate_with_anchor_bias(
        prompt=domain.prompt,
        max_new_tokens=effective_tokens,
        max_length=max_length,
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
    anchor_time = time.time() - t0
    anchor_tokens = len(anchor.get("steps", []))
    print(f"  [ANCHOR] done in {anchor_time:.1f}s, {anchor_tokens} tokens")

    # ── Score ──
    base_analysis = analyze_keywords(
        base["continuation_text"], domain.positive_keywords, domain.negative_keywords,
    )
    anchor_analysis = analyze_keywords(
        anchor.get("continuation_text", ""),
        domain.positive_keywords, domain.negative_keywords,
    )

    # Bias activity
    active_bias_steps = sum(
        1 for step in anchor.get("steps", [])
        if step.get("bias_nonzero_anchors", 0) > 0
    )

    delta_quality = anchor_analysis["quality_score"] - base_analysis["quality_score"]
    anchor_wins = delta_quality > 0
    identical = base["continuation_text"] == anchor.get("continuation_text", "")

    print(f"  base_quality={base_analysis['quality_score']:.1f}  "
          f"anchor_quality={anchor_analysis['quality_score']:.1f}  "
          f"delta={delta_quality:+.1f}  "
          f"{'WIN' if anchor_wins else 'LOSS' if delta_quality < 0 else 'TIE'}")

    return {
        "domain": domain.name,
        "bias_profile": domain.bias_profile_name,
        "prompt": domain.prompt,
        "max_new_tokens": effective_tokens,
        "base_continuation": base["continuation_text"][:2000],
        "anchor_continuation": anchor.get("continuation_text", "")[:2000],
        "base_analysis": base_analysis,
        "anchor_analysis": anchor_analysis,
        "active_bias_steps": active_bias_steps,
        "anchor_total_steps": anchor_tokens,
        "delta_quality": delta_quality,
        "anchor_wins": anchor_wins,
        "identical": identical,
        "base_time_s": round(base_time, 2),
        "anchor_time_s": round(anchor_time, 2),
    }


# ── Summary report ─────────────────────────────────────────────────

def build_summary(results: list[dict[str, Any]], model_name: str) -> str:
    wins = sum(1 for r in results if r["anchor_wins"])
    losses = sum(1 for r in results if r["delta_quality"] < 0)
    ties = sum(1 for r in results if r["delta_quality"] == 0 and not r["identical"])
    identical = sum(1 for r in results if r["identical"])

    lines = [
        "# 20-Domain Retention Campaign Results",
        "",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Domains tested: {len(results)}",
        "",
        "## Score",
        "",
        f"- **ANCHOR wins: {wins}/{len(results)}**",
        f"- BASE wins: {losses}/{len(results)}",
        f"- Ties: {ties}/{len(results)}",
        f"- Identical outputs: {identical}/{len(results)}",
        "",
        "## Per-domain results",
        "",
        "| # | Domain | Profile | Base Q | Anchor Q | Delta | Result |",
        "|---|--------|---------|-------:|---------:|------:|--------|",
    ]
    for i, r in enumerate(results, 1):
        result_tag = "WIN" if r["anchor_wins"] else ("LOSS" if r["delta_quality"] < 0 else "TIE")
        if r["identical"]:
            result_tag = "IDENTICAL"
        lines.append(
            f"| {i} | {r['domain']} | {r['bias_profile']} | "
            f"{r['base_analysis']['quality_score']:.1f} | "
            f"{r['anchor_analysis']['quality_score']:.1f} | "
            f"{r['delta_quality']:+.1f} | {result_tag} |"
        )

    lines.extend([
        "",
        "## Keyword details",
        "",
    ])
    for r in results:
        ba = r["base_analysis"]
        aa = r["anchor_analysis"]
        lines.extend([
            f"### {r['domain']}",
            f"- Base: +{ba['positive_total']} / -{ba['negative_total']} "
            f"(degen={ba['degeneracy_penalty']:.1f})",
            f"- Anchor: +{aa['positive_total']} / -{aa['negative_total']} "
            f"(degen={aa['degeneracy_penalty']:.1f})",
            f"- Bias active steps: {r['active_bias_steps']}/{r['anchor_total_steps']}",
            "",
        ])

    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="20-domain retention campaign")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conflict_threshold", type=float, default=0.55)
    parser.add_argument("--bias_scale", type=float, default=1.50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--frequency_penalty", type=float, default=0.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--max_bias_gate_sum", type=float, default=1.50)
    parser.add_argument("--entropy_top_k", type=int, default=32)
    parser.add_argument("--entropy_threshold", type=float, default=0.35)
    parser.add_argument("--entropy_slope", type=float, default=0.08)
    parser.add_argument("--pressure_threshold", type=float, default=0.60)
    parser.add_argument("--pressure_slope", type=float, default=0.08)
    parser.add_argument("--pressure_rescue_floor", type=float, default=0.20)
    parser.add_argument(
        "--domains", type=str, default="all",
        help="Comma-separated domain names or 'all'",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=ROOT / "archive" / "20domain_campaign",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Select domains
    if args.domains == "all":
        domains = list(RETENTION_DOMAINS)
    else:
        names = {n.strip() for n in args.domains.split(",")}
        domains = [d for d in RETENTION_DOMAINS if d.name in names]
        if not domains:
            print(f"ERROR: no domains matched {names}")
            sys.exit(1)

    print(f"Campaign: {len(domains)} domains on {args.model}")
    print(f"Device: {args.device}")

    # Load model once
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

    # Run all domains
    results: list[dict[str, Any]] = []
    campaign_start = time.time()

    for i, domain in enumerate(domains, 1):
        print(f"\n[{i}/{len(domains)}]", end="")
        try:
            result = run_single_domain(
                overlay, domain,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                conflict_threshold=args.conflict_threshold,
                bias_scale=args.bias_scale,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_bias_gate_sum=args.max_bias_gate_sum,
                entropy_top_k=args.entropy_top_k,
                entropy_threshold=args.entropy_threshold,
                entropy_slope=args.entropy_slope,
                pressure_threshold=args.pressure_threshold,
                pressure_slope=args.pressure_slope,
                pressure_rescue_floor=args.pressure_rescue_floor,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR on {domain.name}: {e}")
            results.append({
                "domain": domain.name,
                "bias_profile": domain.bias_profile_name,
                "error": str(e),
                "anchor_wins": False,
                "delta_quality": 0.0,
                "identical": False,
                "base_analysis": {"quality_score": 0, "positive_total": 0, "negative_total": 0, "degeneracy_penalty": 0},
                "anchor_analysis": {"quality_score": 0, "positive_total": 0, "negative_total": 0, "degeneracy_penalty": 0},
                "active_bias_steps": 0,
                "anchor_total_steps": 0,
            })

    total_time = time.time() - campaign_start

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "n_domains": len(domains),
        "total_time_s": round(total_time, 1),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "conflict_threshold": args.conflict_threshold,
            "bias_scale": args.bias_scale,
            "repetition_penalty": args.repetition_penalty,
            "frequency_penalty": args.frequency_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_bias_gate_sum": args.max_bias_gate_sum,
            "entropy_top_k": args.entropy_top_k,
            "entropy_threshold": args.entropy_threshold,
            "entropy_slope": args.entropy_slope,
            "pressure_threshold": args.pressure_threshold,
            "pressure_slope": args.pressure_slope,
            "pressure_rescue_floor": args.pressure_rescue_floor,
        },
        "summary": {
            "anchor_wins": sum(1 for r in results if r["anchor_wins"]),
            "base_wins": sum(1 for r in results if r["delta_quality"] < 0),
            "ties": sum(1 for r in results if r["delta_quality"] == 0 and not r["identical"]),
            "identical": sum(1 for r in results if r["identical"]),
            "mean_delta_quality": round(
                sum(r["delta_quality"] for r in results) / max(len(results), 1), 3
            ),
        },
        "results": results,
    }

    json_path = args.output_dir / f"campaign_{timestamp}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    md_report = build_summary(results, args.model)
    md_path = args.output_dir / f"campaign_{timestamp}.md"
    md_path.write_text(md_report, encoding="utf-8")

    # Print final summary
    s = payload["summary"]
    print(f"\n{'='*60}")
    print(f"  CAMPAIGN COMPLETE ({total_time:.0f}s)")
    print(f"  ANCHOR wins: {s['anchor_wins']}/{len(results)}")
    print(f"  BASE wins:   {s['base_wins']}/{len(results)}")
    print(f"  Ties:        {s['ties']}/{len(results)}")
    print(f"  Identical:   {s['identical']}/{len(results)}")
    print(f"  Mean Δ quality: {s['mean_delta_quality']:+.3f}")
    print(f"{'='*60}")
    print(f"  JSON: {json_path}")
    print(f"  Report: {md_path}")


if __name__ == "__main__":
    main()
