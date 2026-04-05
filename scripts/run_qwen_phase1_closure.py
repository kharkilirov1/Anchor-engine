"""
ABPT Phase 1 Closure — валидация tail_retention_ratio на всех профилях
======================================================================
Прогоняет phase probe на short, medium, long профилях последовательно.
Сравнивает ρ(tail_retention_ratio, base_constraint_score) между профилями.

Критерий успеха Phase 1: |ρ| > 0.4 на ВСЕХ профилях.

Вывод:
  archive/qwen35_4b_phase1_closure.json
  docs/research/qwen35_4b_phase1_closure.md

Использование:
  python scripts/run_qwen_phase1_closure.py
  python scripts/run_qwen_phase1_closure.py --model Qwen/Qwen3.5-4B --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import (
    ANCHOR_SPAN_PROFILES,
    make_qwen_anchor_geometry_cases,
)
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from scripts.run_qwen_phase_probe import (
    compute_phase_metrics,
    extract_rank1_profile,
    generate_base_text,
    score_constraint,
    spearman_correlation,
    SEED,
)

SUCCESS_THRESHOLD = 0.4


def run_profile(
    overlay: QwenAnchorOverlay,
    profile: str,
    n_layers: int,
    device: torch.device,
) -> dict[str, Any]:
    cases = make_qwen_anchor_geometry_cases(anchor_span_profile=profile)
    probe_layers = list(range(n_layers))

    tail_rets: list[float] = []
    base_scores: list[float] = []
    per_case: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        print(f"  [{profile}] {i+1}/{len(cases)} {case.name}")

        r1_profile = extract_rank1_profile(overlay, case, probe_layers, device)
        if r1_profile is None:
            print(f"    SKIP (span not found)")
            continue

        phase_metrics = compute_phase_metrics(r1_profile, n_layers)
        base_text = generate_base_text(overlay, case.prompt)
        constraint = score_constraint(base_text, case.anchor_group)
        base_score = constraint["constraint_score"]

        tail_ret = phase_metrics.get("tail_retention_ratio")
        if tail_ret is not None:
            tail_rets.append(tail_ret)
            base_scores.append(base_score)

        per_case.append({
            "name": case.name,
            "anchor_group": case.anchor_group,
            "tail_retention_ratio": tail_ret,
            "base_constraint_score": base_score,
            "early_slope_4_8": phase_metrics.get("early_slope_4_8"),
        })

    rho = spearman_correlation(tail_rets, base_scores)
    passed = rho is not None and abs(rho) > SUCCESS_THRESHOLD

    print(f"  [{profile}] ρ = {rho:.4f} | {'PASS' if passed else 'FAIL'}"
          if rho is not None else f"  [{profile}] ρ = N/A | FAIL")

    return {
        "profile": profile,
        "n_cases": len(per_case),
        "rho_tail_retention": rho,
        "passed": passed,
        "per_case": per_case,
    }


def run(model_name: str, device_str: str) -> None:
    torch.manual_seed(SEED)
    device = torch.device(device_str)

    print(f"[Phase1] Model: {model_name}")
    print(f"[Phase1] Device: {device}")
    print(f"[Phase1] Profiles: {ANCHOR_SPAN_PROFILES}")
    print(f"[Phase1] Success criterion: |ρ| > {SUCCESS_THRESHOLD} on ALL profiles\n")

    print("[Phase1] Loading model...")
    overlay = QwenAnchorOverlay.from_pretrained(model_name, config=TOY_CONFIG)
    overlay.to(device)
    overlay.eval()
    n_layers = int(overlay.model_num_hidden_layers)

    profile_results: list[dict[str, Any]] = []
    for profile in ANCHOR_SPAN_PROFILES:
        print(f"\n{'='*60}")
        print(f"  Profile: {profile}")
        print(f"{'='*60}")
        result = run_profile(overlay, profile, n_layers, device)
        profile_results.append(result)

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("PHASE 1 CLOSURE SUMMARY")
    print(f"{'='*60}\n")

    all_passed = True
    comparison: dict[str, Any] = {}
    for pr in profile_results:
        rho = pr["rho_tail_retention"]
        passed = pr["passed"]
        if not passed:
            all_passed = False
        comparison[pr["profile"]] = {
            "rho": rho,
            "passed": passed,
            "n_cases": pr["n_cases"],
        }
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {pr['profile']:8s}: ρ = {rho:.4f} {status}" if rho else
              f"  {pr['profile']:8s}: ρ = N/A   ✗ FAIL")

    phase1_verdict = "CONFIRMED" if all_passed else "NOT_CONFIRMED"
    print(f"\n  Phase 1 verdict: {phase1_verdict}")

    # Check if thresholds are profile-dependent
    rhos = [pr["rho_tail_retention"] for pr in profile_results
            if pr["rho_tail_retention"] is not None]
    if len(rhos) >= 2:
        rho_spread = max(rhos) - min(rhos)
        transferable = rho_spread < 0.3
        print(f"  ρ spread across profiles: {rho_spread:.4f} "
              f"({'transferable' if transferable else 'profile-dependent'})")
    else:
        transferable = None
        rho_spread = None

    # ─────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────

    ARCHIVE = ROOT / "archive"
    ARCHIVE.mkdir(exist_ok=True)
    slug = model_name.split("/")[-1].lower().replace("-", "_").replace(".", "")

    payload = {
        "metadata": {
            "model_name": model_name,
            "n_layers": n_layers,
            "profiles_tested": list(ANCHOR_SPAN_PROFILES),
            "success_threshold": SUCCESS_THRESHOLD,
            "seed": SEED,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        "verdict": phase1_verdict,
        "comparison": comparison,
        "rho_spread": rho_spread,
        "transferable": transferable,
        "profile_results": profile_results,
    }

    out_json = ARCHIVE / f"{slug}_phase1_closure.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[Phase1] JSON → {out_json}")

    # --- MD report ---
    DOCS = ROOT / "docs" / "research"
    DOCS.mkdir(parents=True, exist_ok=True)
    out_md = DOCS / f"{slug}_phase1_closure.md"

    md = [
        f"# Phase 1 Closure — {model_name}",
        f"",
        f"**Created:** {payload['metadata']['created_at_utc']}",
        f"**Verdict:** {phase1_verdict}",
        f"**Criterion:** |ρ(tail_retention_ratio, base_constraint)| > {SUCCESS_THRESHOLD} on all profiles",
        f"",
        f"## Results",
        f"",
        f"| Profile | ρ | Cases | Status |",
        f"|---------|---|-------|--------|",
    ]
    for pr in profile_results:
        rho = pr["rho_tail_retention"]
        status = "PASS" if pr["passed"] else "FAIL"
        md.append(f"| {pr['profile']} | {rho:.4f if rho else 'N/A'} | {pr['n_cases']} | {status} |")

    if rho_spread is not None:
        md += [
            f"",
            f"## Transferability",
            f"",
            f"ρ spread: {rho_spread:.4f} — {'thresholds transferable' if transferable else 'thresholds are profile-dependent'}",
        ]

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[Phase1] MD   → {out_md}")
    print(f"\n[Phase1] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ABPT Phase 1 Closure")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args()
    run(model_name=args.model, device_str=args.device)


if __name__ == "__main__":
    main()
