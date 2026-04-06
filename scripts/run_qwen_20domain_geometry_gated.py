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
from datetime import datetime, timezone
import json
import os
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
    auto_calibrate_thresholds,
    build_tail_reference_layers,
    compute_geometry_metrics,
    detect_anchor_span,
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
    """Run geometry probe using attention-based anchor detection.

    Flow: forward pass (with attentions) → detect anchor span from
    attention mass → extract geometry at detected span.
    Falls back to text-based match_anchor_span if attentions unavailable.
    """
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        return {"matched": False, "cluster": "unknown", "reason": "no_tokenizer"}

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

    # Forward pass FIRST — need attentions for anchor detection
    input_tensor = encoded["input_ids"].to(device)
    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=input_tensor,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    # Robust attention extraction: some model wrappers nest attentions
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        attentions = getattr(outputs, "decoder_attentions", None)
    if attentions is None and hasattr(outputs, "language_model_outputs"):
        attentions = getattr(outputs.language_model_outputs, "attentions", None)

    # Primary: attention-based anchor detection
    span_match = None
    if attentions is not None:
        span_match = detect_anchor_span(
            attentions,
            probe_layers,
            min_width=5,
            max_width=10,
        )

    # Fallback: text-based matching (for models without attention output)
    if span_match is None and domain.anchor_text:
        offsets = None
        if "offset_mapping" in encoded:
            offsets = [tuple(pair) for pair in encoded["offset_mapping"][0].tolist()]
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
            "reason": "no_anchor_detected",
        }

    # Ensure span is wide enough for meaningful geometry (need ≥4 tokens for r1)
    seq_len = len(input_ids)
    MIN_GEOMETRY_TOKENS = 5
    if span_match.token_count < MIN_GEOMETRY_TOKENS:
        center = (span_match.token_start + span_match.token_end) // 2
        half = MIN_GEOMETRY_TOKENS // 2
        new_start = max(1, center - half)  # skip BOS
        new_end = min(seq_len - 2, new_start + MIN_GEOMETRY_TOKENS - 1)
        new_start = max(1, new_end - MIN_GEOMETRY_TOKENS + 1)
        span_match = AnchorSpanMatch(
            anchor_text=span_match.anchor_text,
            token_start=new_start,
            token_end=new_end,
            token_count=new_end - new_start + 1,
            char_start=span_match.char_start,
            char_end=span_match.char_end,
            match_method=f"{span_match.match_method}+expanded",
            matched_text=span_match.matched_text,
        )

    # Decode detected span for logging
    detected_text = ""
    try:
        detected_text = tokenizer.decode(
            input_ids[span_match.token_start : span_match.token_end + 1],
            skip_special_tokens=True,
        )
    except Exception:
        pass

    # Extract r1 profile across probe layers
    r1_profile: dict[str, float | None] = {}
    for layer_idx in probe_layers:
        hs_idx = layer_idx + 1
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
        route = "base"
    elif delta_template is not None and delta_template > template_delta_threshold:
        cluster = "template"
        route = "base"
    else:
        cluster = "flat"
        route = "anchor"

    return {
        "matched": True,
        "cluster": cluster,
        "route": route,
        "r1_reference": r1_ref,
        "delta_template_pair": delta_template,
        "mature_layer": mature_layer,
        "anchor_text": domain.anchor_text,
        "detected_text": detected_text,
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

    # H6: continuous bias — scale strength by geometry instead of binary gate
    r1_ref = geometry.get("r1_reference")
    r1_ceiling = geometry.get("r1_ceiling", 0.40)
    if r1_ref is not None:
        continuous_scale = bias_scale * max(0.0, 1.0 - r1_ref / r1_ceiling)
    else:
        continuous_scale = bias_scale  # no geometry → full bias

    if continuous_scale > 0.01:
        t0 = time.time()
        print(f"  [ANCHOR] generating (continuous bias_scale={continuous_scale:.3f})...")
        anchor = overlay.generate_with_anchor_bias(
            prompt=domain.prompt,
            max_new_tokens=effective_tokens,
            max_length=max_length,
            conflict_threshold=conflict_threshold,
            bias_scale=continuous_scale,
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
        print(f"  [SKIP ANCHOR] bias_scale={continuous_scale:.4f} ≈ 0 ({cluster})")
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
        "continuous_bias_scale": round(continuous_scale, 4),
        "r1_reference": geometry.get("r1_reference"),
        "r1_ceiling": r1_ceiling,
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
        attn_implementation=os.environ.get("ATTN_IMPL", "eager"),  # eager for attentions, sdpa for VRAM savings
    )
    overlay.eval()

    # Setup geometry probing
    n_layers = int(overlay.model_num_hidden_layers)
    probe_layers = select_tail_probe_layers(n_layers, count=10)
    reference_layers = build_tail_reference_layers(probe_layers)

    print(f"Probe layers: {probe_layers}")
    print(f"Reference: mature=L{reference_layers['mature_layer']}, "
          f"template=L{reference_layers['template_prev_layer']}-L{reference_layers['template_curr_layer']}")

    # ── Phase 1: Geometry probing + auto-calibration ─────────────────
    print(f"\n{'='*60}")
    print("  PHASE 1a: Geometry probing (collect metrics)")
    print(f"{'='*60}")

    # Pass 1: collect raw r1 and delta_template for all domains
    geometries: list[dict[str, Any]] = []
    for domain in domains:
        geo = probe_geometry(
            overlay, domain, probe_layers, reference_layers,
            mature_r1_threshold=args.mature_r1_threshold,
            template_delta_threshold=args.template_delta_threshold,
        )
        geometries.append(geo)
        r1 = geo.get("r1_reference")
        r1_s = f"{r1:.3f}" if r1 is not None else "N/A"
        dt = geo.get("delta_template_pair")
        dt_s = f"{dt:+.4f}" if dt is not None else "N/A"
        method = geo.get("match_method", geo.get("reason", "?"))
        print(f"  {domain.name:<35} r1={r1_s:<8} delta_tpl={dt_s:<10} match={method}")

    # Pass 2: auto-calibrate thresholds from observed distribution
    r1_values = [g["r1_reference"] for g in geometries if g.get("r1_reference") is not None]
    dt_values = [g["delta_template_pair"] for g in geometries if g.get("delta_template_pair") is not None]

    calibration: dict[str, Any] = {}
    if len(r1_values) >= 3 and len(dt_values) == len(r1_values):
        calibration = auto_calibrate_thresholds(r1_values, dt_values)
        cal_r1 = calibration["mature_r1_threshold"]
        cal_dt = calibration["template_delta_threshold"]
        print(f"\n  Auto-calibrated thresholds (k-means on {calibration['n_samples']} samples):")
        print(f"    mature_r1_threshold:    {args.mature_r1_threshold:.3f} → {cal_r1:.3f}")
        print(f"    template_delta_threshold: {args.template_delta_threshold:.4f} → {cal_dt:.4f}")
        for cc in calibration.get("cluster_centers", []):
            print(f"    cluster '{cc['label']}': r1={cc['r1']:.3f}, delta={cc['delta']:+.4f}")
    else:
        cal_r1 = args.mature_r1_threshold
        cal_dt = args.template_delta_threshold
        print(f"\n  Not enough data for auto-calibration ({len(r1_values)} samples), using defaults")

    # Pass 3: reclassify with calibrated thresholds
    print(f"\n{'='*60}")
    print("  PHASE 1b: Classification (calibrated thresholds)")
    print(f"{'='*60}")

    # r1_ceiling for H6 continuous bias: max observed r1 (at ceiling → bias=0)
    r1_ceiling = max(r1_values) if r1_values else 0.40

    for geo in geometries:
        if not geo.get("matched"):
            continue
        r1_ref = geo.get("r1_reference")
        delta_tpl = geo.get("delta_template_pair")
        geo["r1_ceiling"] = r1_ceiling
        if r1_ref is not None and r1_ref > cal_r1:
            geo["cluster"] = "mature"
            geo["route"] = "base"
        elif delta_tpl is not None and delta_tpl > cal_dt:
            geo["cluster"] = "template"
            geo["route"] = "base"
        else:
            geo["cluster"] = "flat"
            geo["route"] = "anchor"

    print(f"  r1_ceiling (max observed): {r1_ceiling:.3f}")
    print(f"  bias_scale formula: {args.bias_scale:.2f} * max(0, 1 - r1/{r1_ceiling:.3f})")
    print()

    for domain, geo in zip(domains, geometries):
        cluster = geo.get("cluster", "?")
        r1 = geo.get("r1_reference")
        r1_s = f"{r1:.3f}" if r1 is not None else "N/A"
        if r1 is not None:
            cs = args.bias_scale * max(0.0, 1.0 - r1 / r1_ceiling)
        else:
            cs = args.bias_scale
        print(f"  {domain.name:<35} cluster={cluster:<10} r1={r1_s:<8} bias={cs:.3f}")

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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "n_domains": len(domains),
        "total_time_s": round(total_time, 1),
        "thresholds": {
            "mature_r1_threshold_default": args.mature_r1_threshold,
            "template_delta_threshold_default": args.template_delta_threshold,
            "mature_r1_threshold_calibrated": cal_r1,
            "template_delta_threshold_calibrated": cal_dt,
            "calibration_method": calibration.get("method", "fallback_default"),
        },
        "calibration": calibration,
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
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
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
