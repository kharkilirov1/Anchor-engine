"""Geometry-gated 20-domain retention campaign.

For each domain:
1. Run geometry probe → classify cluster (mature/template/flat)
2. If flat → use ANCHOR bias (model needs help)
3. If mature/template → use BASE (model can handle it)
4. Compare with ungated results (always-anchor)

Usage:
    python scripts/run_qwen_20domain_geometry_gated.py --model Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
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
from src.utils.anchor_geometry import (
    AnchorSpanMatch,
    build_tail_reference_layers,
    compute_geometry_metrics,
    extract_delta_vectors,
    match_anchor_span,
    select_tail_probe_layers,
)
from scripts.run_qwen_20domain_retention_campaign import (
    analyze_keywords,
    generate_base,
)


# ── Geometry probe for a single domain ──────────────────────────────

def probe_geometry(
    overlay: QwenAnchorOverlay,
    domain: RetentionDomain,
    probe_layers: list[int],
    reference_layers: dict[str, int],
    *,
    max_length: int = 128,
    mature_r1_threshold: float = 0.65,
    template_delta_threshold: float = 0.08,
) -> dict[str, Any]:
    """Run geometry probe on domain's anchor span, return cluster classification."""
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        return {"matched": False, "cluster": "unknown", "reason": "no_tokenizer"}

    if not domain.anchor_text:
        return {"matched": False, "cluster": "unknown", "reason": "no_anchor_text"}

    device = next(overlay.parameters()).device

    # Tokenize prompt
    encoded = tokenizer(
        domain.prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"][0].tolist()
    offsets = None
    if "offset_mapping" in encoded:
        offsets = [tuple(pair) for pair in encoded["offset_mapping"][0].tolist()]

    # Match anchor span in tokens
    span_match = match_anchor_span(
        text=domain.prompt,
        anchor_text=domain.anchor_text,
        input_ids=input_ids,
        tokenizer=tokenizer,
        offsets=offsets,
    )
    if span_match is None:
        return {
            "matched": False,
            "cluster": "unknown",
            "reason": f"span_not_matched: '{domain.anchor_text}'",
        }

    # Run forward pass to get hidden states
    input_tensor = encoded["input_ids"].to(device)
    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=input_tensor,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors

    # Extract r1 profile across probe layers
    r1_profile: dict[str, float | None] = {}
    for layer_idx in probe_layers:
        hs_idx = layer_idx + 1  # hidden_states[0] = embeddings
        if hs_idx >= len(hidden_states):
            r1_profile[str(layer_idx)] = None
            continue
        hs = hidden_states[hs_idx][0]  # [seq_len, hidden_dim]
        try:
            delta_vecs = extract_delta_vectors(hs, span_match.token_start, span_match.token_end)
            metrics = compute_geometry_metrics(delta_vecs)
            r1_profile[str(layer_idx)] = metrics.get("rank1_explained_variance")
        except (ValueError, RuntimeError):
            r1_profile[str(layer_idx)] = None

    # Classify cluster
    mature_layer = reference_layers["mature_layer"]
    template_prev = reference_layers["template_prev_layer"]
    template_curr = reference_layers["template_curr_layer"]

    r1_ref = r1_profile.get(str(mature_layer))
    r1_prev = r1_profile.get(str(template_prev))
    r1_curr = r1_profile.get(str(template_curr))

    delta_template = None
    if r1_prev is not None and r1_curr is not None:
        delta_template = r1_curr - r1_prev

    if r1_ref is not None and r1_ref > mature_r1_threshold:
        cluster = "mature"
        route = "base"  # model can handle it
    elif delta_template is not None and delta_template > template_delta_threshold:
        cluster = "template"
        route = "base"  # model knows the pattern
    else:
        cluster = "flat"
        route = "anchor"  # model needs help

    return {
        "matched": True,
        "cluster": cluster,
        "route": route,
        "r1_reference": r1_ref,
        "delta_template_pair": delta_template,
        "mature_layer": mature_layer,
        "anchor_text": domain.anchor_text,
        "match_method": span_match.match_method,
        "token_start": span_match.token_start,
        "token_end": span_match.token_end,
        "token_count": span_match.token_count,
        "r1_profile": r1_profile,
    }


# ── Single domain run with geometry gating ──────────────────────────

def run_gated_domain(
    overlay: QwenAnchorOverlay,
    domain: RetentionDomain,
    geometry: dict[str, Any],
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
    effective_tokens = domain.max_new_tokens or max_new_tokens
    cluster = geometry.get("cluster", "unknown")
    route = geometry.get("route", "anchor")  # default to anchor if unknown

    print(f"\n{'='*60}")
    print(f"  Domain: {domain.name}")
    print(f"  Cluster: {cluster} | Route: {route}")
    print(f"  r1_ref: {geometry.get('r1_reference', '?')} | "
          f"delta_tpl: {geometry.get('delta_template_pair', '?')}")
    print(f"{'='*60}")

    # Always run BASE
    t0 = time.time()
    print("  [BASE] generating...")
    base = generate_base(
        overlay, domain.prompt,
        max_new_tokens=effective_tokens, max_length=max_length,
    )
    base_time = time.time() - t0
    print(f"  [BASE] done in {base_time:.1f}s")

    # Run ANCHOR only if route says so
    if route == "anchor":
        t0 = time.time()
        print("  [ANCHOR] generating (geometry says: INTERVENE)...")
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
        anchor_text = anchor.get("continuation_text", "")
        anchor_tokens = len(anchor.get("steps", []))
        active_bias = sum(
            1 for s in anchor.get("steps", []) if s.get("bias_nonzero_anchors", 0) > 0
        )
        print(f"  [ANCHOR] done in {anchor_time:.1f}s")
        chosen = "anchor"
    else:
        print(f"  [SKIP ANCHOR] geometry says: model can handle it ({cluster})")
        anchor_text = base["continuation_text"]
        anchor_tokens = base["num_tokens"]
        anchor_time = 0.0
        active_bias = 0
        chosen = "base"

    # Score
    base_analysis = analyze_keywords(
        base["continuation_text"], domain.positive_keywords, domain.negative_keywords,
    )
    chosen_text = anchor_text if chosen == "anchor" else base["continuation_text"]
    chosen_analysis = analyze_keywords(
        chosen_text, domain.positive_keywords, domain.negative_keywords,
    )

    delta = chosen_analysis["quality_score"] - base_analysis["quality_score"]
    gated_wins = delta > 0

    result_tag = "WIN" if gated_wins else ("SAME" if chosen == "base" else "LOSS")
    print(f"  chosen={chosen}  base_q={base_analysis['quality_score']:.1f}  "
          f"chosen_q={chosen_analysis['quality_score']:.1f}  "
          f"delta={delta:+.1f}  {result_tag}")

    return {
        "domain": domain.name,
        "bias_profile": domain.bias_profile_name,
        "cluster": cluster,
        "route": route,
        "chosen": chosen,
        "r1_reference": geometry.get("r1_reference"),
        "delta_template_pair": geometry.get("delta_template_pair"),
        "base_continuation": base["continuation_text"][:2000],
        "chosen_continuation": chosen_text[:2000],
        "base_analysis": base_analysis,
        "chosen_analysis": chosen_analysis,
        "active_bias_steps": active_bias,
        "total_steps": anchor_tokens,
        "delta_quality": delta,
        "gated_wins": gated_wins,
        "base_time_s": round(base_time, 2),
        "anchor_time_s": round(anchor_time, 2),
        "geometry": {
            k: v for k, v in geometry.items()
            if k != "r1_profile"  # save space
        },
    }


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-gated 20-domain campaign")
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
    parser.add_argument("--mature_r1_threshold", type=float, default=0.65)
    parser.add_argument("--template_delta_threshold", type=float, default=0.08)
    parser.add_argument("--domains", type=str, default="all")
    parser.add_argument(
        "--output_dir", type=Path,
        default=ROOT / "archive" / "20domain_geometry_gated",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.domains == "all":
        domains = list(RETENTION_DOMAINS)
    else:
        names = {n.strip() for n in args.domains.split(",")}
        domains = [d for d in RETENTION_DOMAINS if d.name in names]

    print(f"Geometry-gated campaign: {len(domains)} domains on {args.model}")

    # Load model
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

    # Setup geometry probing
    n_layers = int(overlay.model_num_hidden_layers)
    probe_layers = select_tail_probe_layers(n_layers, count=10)
    reference_layers = build_tail_reference_layers(probe_layers)

    print(f"Probe layers: {probe_layers}")
    print(f"Reference: mature=L{reference_layers['mature_layer']}, "
          f"template=L{reference_layers['template_prev_layer']}-L{reference_layers['template_curr_layer']}")

    # ── Phase 1: Geometry classification ────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 1: Geometry classification")
    print(f"{'='*60}")

    geometries: list[dict[str, Any]] = []
    for domain in domains:
        geo = probe_geometry(
            overlay, domain, probe_layers, reference_layers,
            mature_r1_threshold=args.mature_r1_threshold,
            template_delta_threshold=args.template_delta_threshold,
        )
        geometries.append(geo)
        cluster = geo.get("cluster", "?")
        route = geo.get("route", "?")
        r1 = geo.get("r1_reference")
        r1_s = f"{r1:.3f}" if r1 is not None else "N/A"
        method = geo.get("match_method", geo.get("reason", "?"))
        print(f"  {domain.name:<35} cluster={cluster:<10} r1={r1_s:<8} route={route:<8} match={method}")

    n_flat = sum(1 for g in geometries if g.get("cluster") == "flat")
    n_mature = sum(1 for g in geometries if g.get("cluster") == "mature")
    n_template = sum(1 for g in geometries if g.get("cluster") == "template")
    n_unknown = sum(1 for g in geometries if g.get("cluster") == "unknown")

    print(f"\n  Clusters: flat={n_flat}, mature={n_mature}, "
          f"template={n_template}, unknown={n_unknown}")
    print(f"  Will run ANCHOR on: {n_flat + n_unknown} domains")
    print(f"  Will skip ANCHOR on: {n_mature + n_template} domains")

    # ── Phase 2: Gated generation ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 2: Gated generation")
    print(f"{'='*60}")

    results: list[dict[str, Any]] = []
    campaign_start = time.time()

    for i, (domain, geo) in enumerate(zip(domains, geometries), 1):
        print(f"\n[{i}/{len(domains)}]", end="")
        try:
            result = run_gated_domain(
                overlay, domain, geo,
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
            print(f"  ERROR: {e}")
            results.append({
                "domain": domain.name,
                "cluster": geo.get("cluster", "unknown"),
                "route": geo.get("route", "anchor"),
                "chosen": "error",
                "error": str(e),
                "delta_quality": 0.0,
                "gated_wins": False,
            })

    total_time = time.time() - campaign_start

    # ── Summary ────────────────────────────────────────────────────
    anchor_used = [r for r in results if r.get("chosen") == "anchor"]
    base_used = [r for r in results if r.get("chosen") == "base"]
    anchor_wins = sum(1 for r in anchor_used if r.get("gated_wins"))
    anchor_losses = sum(1 for r in anchor_used if r.get("delta_quality", 0) < 0)
    total_losses = sum(1 for r in results if r.get("delta_quality", 0) < 0)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "n_domains": len(domains),
        "total_time_s": round(total_time, 1),
        "thresholds": {
            "mature_r1_threshold": args.mature_r1_threshold,
            "template_delta_threshold": args.template_delta_threshold,
        },
        "geometry_summary": {
            "flat": n_flat,
            "mature": n_mature,
            "template": n_template,
            "unknown": n_unknown,
        },
        "gating_summary": {
            "anchor_invoked": len(anchor_used),
            "base_kept": len(base_used),
            "anchor_wins": anchor_wins,
            "anchor_losses": anchor_losses,
            "total_losses": total_losses,
            "gating_efficiency": f"{anchor_wins}/{len(anchor_used)}" if anchor_used else "N/A",
        },
        "results": results,
    }

    json_path = args.output_dir / f"gated_campaign_{timestamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown
    lines = [
        "# Geometry-Gated 20-Domain Campaign",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{args.model}`",
        f"Thresholds: mature_r1>{args.mature_r1_threshold}, "
        f"template_delta>{args.template_delta_threshold}",
        "",
        "## Geometry clusters",
        "",
        f"- flat (anchor needed): **{n_flat}**",
        f"- mature (model ok): **{n_mature}**",
        f"- template (model ok): **{n_template}**",
        f"- unknown: {n_unknown}",
        "",
        "## Gating results",
        "",
        f"- Anchor invoked: **{len(anchor_used)}/{len(results)}** domains",
        f"- Anchor wins: **{anchor_wins}/{len(anchor_used)}**"
        if anchor_used else "- Anchor wins: N/A",
        f"- Anchor losses: {anchor_losses}/{len(anchor_used)}"
        if anchor_used else "- Anchor losses: N/A",
        f"- **Total LOSS count: {total_losses}** "
        f"(vs 14 without gating)",
        "",
        "## Per-domain",
        "",
        "| # | Domain | Cluster | Route | Chosen | Base Q | Chosen Q | Delta | Result |",
        "|---|--------|---------|-------|--------|-------:|---------:|------:|--------|",
    ]
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"| {i} | {r['domain']} | {r.get('cluster','?')} | "
                         f"{r.get('route','?')} | ERROR | | | | {r['error'][:20]} |")
            continue
        bq = r.get("base_analysis", {}).get("quality_score", 0)
        cq = r.get("chosen_analysis", {}).get("quality_score", 0)
        tag = "WIN" if r.get("gated_wins") else ("SAME" if r["chosen"] == "base" and r["delta_quality"] == 0 else ("LOSS" if r["delta_quality"] < 0 else "SAME"))
        lines.append(
            f"| {i} | {r['domain']} | {r['cluster']} | {r['route']} | "
            f"{r['chosen']} | {bq:.1f} | {cq:.1f} | {r['delta_quality']:+.1f} | {tag} |"
        )

    lines.extend(["", ""])
    md_path = args.output_dir / f"gated_campaign_{timestamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  GEOMETRY-GATED CAMPAIGN COMPLETE ({total_time:.0f}s)")
    print(f"  Clusters: flat={n_flat}, mature={n_mature}, template={n_template}")
    print(f"  Anchor invoked: {len(anchor_used)}/{len(results)}")
    print(f"  Anchor wins: {anchor_wins}/{len(anchor_used)}" if anchor_used else "  No anchor invocations")
    print(f"  Total LOSS: {total_losses} (was 14 without gating)")
    print(f"{'='*60}")
    print(f"  JSON: {json_path}")
    print(f"  Report: {md_path}")


if __name__ == "__main__":
    main()
