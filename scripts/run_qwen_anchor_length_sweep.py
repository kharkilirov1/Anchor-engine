from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.data.qwen_anchor_geometry_cases import (
    list_anchor_span_profiles,
    make_qwen_anchor_geometry_cases,
)
from scripts.run_qwen_geometry_generation_calibration import (
    DEFAULT_BIAS_SCALE,
    DEFAULT_CONFLICT_THRESHOLD,
    DEFAULT_ENTROPY_SLOPE,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_ENTROPY_TOP_K,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_BIAS_GATE_SUM,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_PRESSURE_RESCUE_FLOOR,
    DEFAULT_PRESSURE_SLOPE,
    DEFAULT_PRESSURE_THRESHOLD,
    DEFAULT_REPETITION_PENALTY,
    analyze_case,
    resolve_geometry_probe_layers,
    search_reference_and_thresholds,
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _policy_rank_key(policy_stats: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(policy_stats.get("delta_vs_always_base") or 0.0),
        -float(policy_stats.get("losses_vs_base") or 0.0),
        float(policy_stats.get("wins_over_base") or 0.0),
    )


def _best_policy_name(policy_simulation: dict[str, Any]) -> str:
    all_cases = dict(policy_simulation.get("all_cases", {}))
    if not all_cases:
        return "n/a"
    return max(all_cases.keys(), key=lambda name: _policy_rank_key(dict(all_cases[name])))


def _token_count_stats(cases: list[dict[str, Any]]) -> dict[str, float | int | None]:
    token_counts = [
        int(dict(case.get("span_match", {})).get("token_count"))
        for case in cases
        if dict(case.get("span_match", {})).get("token_count") is not None
    ]
    if not token_counts:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    values = np.array(token_counts, dtype=np.float64)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def _profile_summary(
    *,
    profile_name: str,
    cases: list[dict[str, Any]],
    search_result: dict[str, Any],
) -> dict[str, Any]:
    token_stats = _token_count_stats(cases)
    calibration = dict(search_result["calibration"])
    policy_simulation = dict(search_result["policy_simulation"])
    best_candidate = dict(search_result["candidate"])
    best_policy = _best_policy_name(policy_simulation)
    all_policy_stats = dict(policy_simulation.get("all_cases", {})).get(best_policy, {})
    return {
        "profile": profile_name,
        "n_cases": len(cases),
        "token_count_stats": token_stats,
        "best_candidate": best_candidate,
        "cluster_counts": dict(best_candidate.get("cluster_counts", {})),
        "best_policy": best_policy,
        "best_policy_stats": dict(all_policy_stats),
        "clean_base_observed_separation": bool(
            dict(calibration.get("threshold_candidates", {})).get("clean_base_observed_separation", False)
        ),
        "flat_cluster_summary": dict(calibration.get("by_cluster_clean_base", {})).get("flat", {}),
        "template_cluster_summary": dict(calibration.get("by_cluster_clean_base", {})).get("template", {}),
        "mature_cluster_summary": dict(calibration.get("by_cluster_clean_base", {})).get("mature", {}),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def build_length_sweep_markdown(
    *,
    model_name: str,
    device: str,
    profile_summaries: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen Anchor Length Sweep",
        "",
        "## Summary",
        "",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        f"- Profiles: `{[summary['profile'] for summary in profile_summaries]}`",
        "",
        "## Length-to-quality overview",
        "",
        "| profile | mean_token_count | min | max | best_policy | delta_vs_base | wins | losses | clean_sep | cluster_counts |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in profile_summaries:
        best_policy_stats = dict(summary["best_policy_stats"])
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["profile"],
                    _fmt(summary["token_count_stats"]["mean"]),
                    _fmt(summary["token_count_stats"]["min"]),
                    _fmt(summary["token_count_stats"]["max"]),
                    summary["best_policy"],
                    _fmt(best_policy_stats.get("delta_vs_always_base")),
                    str(int(best_policy_stats.get("wins_over_base") or 0)),
                    str(int(best_policy_stats.get("losses_vs_base") or 0)),
                    str(summary["clean_base_observed_separation"]),
                    f"`{summary['cluster_counts']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best searched configuration per profile",
            "",
            "| profile | reference_layers | thresholds | flat_mean_delta | flat_rescue_rate |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for summary in profile_summaries:
        best_candidate = dict(summary["best_candidate"])
        flat_summary = dict(summary["flat_cluster_summary"])
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["profile"],
                    f"`{best_candidate.get('reference_layers', {})}`",
                    f"`{best_candidate.get('thresholds', {})}`",
                    _fmt(flat_summary.get("mean_constraint_delta")),
                    _fmt(flat_summary.get("rescue_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Compare the mean matched token count with policy delta and flat-cluster behavior. "
                "If medium spans keep or improve routing utility while long spans only improve geometric separation, "
                "that is evidence for an anchor-length sweet spot rather than monotonic benefit from longer spans."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep anchor span length profiles for Qwen geometry calibration.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(list_anchor_span_profiles()),
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_anchor_length_sweep.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_length_sweep.md",
    )
    args = parser.parse_args()

    for profile in args.profiles:
        if profile not in list_anchor_span_profiles():
            raise ValueError(f"unknown profile: {profile}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        low_cpu_mem_usage=True,
    )
    overlay.eval()
    device = torch.device(args.device)
    probe_layers, search_layers = resolve_geometry_probe_layers(overlay)

    profile_payloads: list[dict[str, Any]] = []
    profile_summaries: list[dict[str, Any]] = []

    for profile_name in args.profiles:
        cases = make_qwen_anchor_geometry_cases(anchor_span_profile=profile_name)
        if args.case_name:
            cases = [case for case in cases if case.name == args.case_name]
        if args.limit is not None:
            cases = cases[: max(int(args.limit), 0)]
        if not cases:
            raise ValueError(f"no cases selected for profile {profile_name}")
        records: list[dict[str, Any]] = []
        for case in cases:
            record = analyze_case(
                overlay=overlay,
                case=case,
                probe_layers=probe_layers,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                conflict_threshold=DEFAULT_CONFLICT_THRESHOLD,
                bias_scale=DEFAULT_BIAS_SCALE,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
                no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                max_bias_gate_sum=DEFAULT_MAX_BIAS_GATE_SUM,
                entropy_top_k=DEFAULT_ENTROPY_TOP_K,
                entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
                entropy_slope=DEFAULT_ENTROPY_SLOPE,
                pressure_threshold=DEFAULT_PRESSURE_THRESHOLD,
                pressure_slope=DEFAULT_PRESSURE_SLOPE,
                pressure_rescue_floor=DEFAULT_PRESSURE_RESCUE_FLOOR,
                device=device,
            )
            if record is not None:
                records.append(record)
        search_result = search_reference_and_thresholds(records, search_layers=search_layers)
        profile_payload = {
            "profile": profile_name,
            "cases": search_result["cases"],
            "calibration": search_result["calibration"],
            "policy_simulation": search_result["policy_simulation"],
            "search": search_result["search_summary"],
        }
        profile_payloads.append(profile_payload)
        profile_summaries.append(
            _profile_summary(
                profile_name=profile_name,
                cases=search_result["cases"],
                search_result=search_result,
            )
        )

    profile_summaries.sort(key=lambda item: _to_float(item["token_count_stats"]["mean"]) or -1.0)
    best_profile = max(
        profile_summaries,
        key=lambda item: _policy_rank_key(dict(item["best_policy_stats"])),
    )
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": args.model,
            "device": args.device,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "probe_layers": probe_layers,
            "search_layers": search_layers,
            "profiles": args.profiles,
            "seed": args.seed,
        },
        "curve_points": [
            {
                "profile": summary["profile"],
                "mean_token_count": summary["token_count_stats"]["mean"],
                "best_policy": summary["best_policy"],
                "delta_vs_base": dict(summary["best_policy_stats"]).get("delta_vs_always_base"),
                "wins": dict(summary["best_policy_stats"]).get("wins_over_base"),
                "losses": dict(summary["best_policy_stats"]).get("losses_vs_base"),
                "flat_mean_constraint_delta": dict(summary["flat_cluster_summary"]).get("mean_constraint_delta"),
                "flat_rescue_rate": dict(summary["flat_cluster_summary"]).get("rescue_rate"),
                "clean_base_observed_separation": summary["clean_base_observed_separation"],
            }
            for summary in profile_summaries
        ],
        "best_profile": best_profile,
        "profiles": profile_payloads,
    }
    report = build_length_sweep_markdown(
        model_name=args.model,
        device=args.device,
        profile_summaries=profile_summaries,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
