from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import (
    list_anchor_span_profiles,
    make_qwen_anchor_geometry_cases,
)
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import list_model_layers
from scripts.run_qwen_anchor_geometry_probe import (
    aggregate_results,
    analyze_case_geometry,
    infer_overall_interpretation,
    strip_mean_directions,
)


SUPPORT_ORDER = {
    "not_supported": 0,
    "partially_supported": 1,
    "supported": 2,
}

VERDICT_ORDER = {
    "no_signal": 0,
    "mixed_signal": 1,
    "partial_separation": 2,
    "clear_separation": 3,
}


def _token_count_stats(cases: list[Any]) -> dict[str, float | int | None]:
    token_counts = [len(str(case.anchor_text).split()) for case in cases]
    if not token_counts:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": float(statistics.fmean(token_counts)),
        "median": float(statistics.median(token_counts)),
    }


def _layer_value(value: Any) -> int | None:
    if isinstance(value, dict):
        layer = value.get("layer")
        if layer is None:
            return None
        return int(layer)
    if value is None:
        return None
    return int(value)


def _group_transition_table(mode_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for group_name, group_entry in sorted(dict(mode_result.get("group_aggregates", {})).items()):
        transition = dict(group_entry.get("transition_summary", {}))
        summary[group_name] = {
            "anchor_class": group_entry.get("anchor_class"),
            "transitional": bool(transition.get("transitional", False)),
            "sign_changes": int(transition.get("sign_changes", 0) or 0),
            "first_content_like_layer": _layer_value(transition.get("first_content_like_layer")),
            "first_procedure_like_layer": _layer_value(transition.get("first_procedure_like_layer")),
            "strongest_content_like_layer": _layer_value(transition.get("strongest_content_like_layer")),
            "strongest_procedure_like_layer": _layer_value(transition.get("strongest_procedure_like_layer")),
        }
    return summary


def _case_to_dict(case: Any) -> dict[str, Any]:
    if is_dataclass(case):
        return asdict(case)
    if hasattr(case, "__dict__"):
        return dict(vars(case))
    raise TypeError(f"unsupported case type: {type(case)!r}")


def _profile_rank_key(summary: dict[str, Any]) -> tuple[int, int, int, int]:
    interpretation = dict(summary.get("interpretation", {}))
    trimmed_clean = dict(summary.get("trimmed_clean", {}))
    stable_birth = trimmed_clean.get("stable_birth_layer")
    first_positive = trimmed_clean.get("first_positive_layer")
    return (
        SUPPORT_ORDER.get(str(interpretation.get("support_after_tokenization_controls")), -1),
        VERDICT_ORDER.get(str(trimmed_clean.get("verdict")), -1),
        -int(stable_birth) if stable_birth is not None else -999,
        -int(first_positive) if first_positive is not None else -999,
    )


def summarize_profile(
    *,
    profile_name: str,
    cases: list[Any],
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    interpretation: dict[str, Any],
) -> dict[str, Any]:
    full_clean = dict(aggregate["modes"]["full_span"]["clean_only"])
    trimmed_clean = dict(aggregate["modes"]["trimmed_span"]["clean_only"])
    full_all = dict(aggregate["modes"]["full_span"]["all_valid"])
    trimmed_all = dict(aggregate["modes"]["trimmed_span"]["all_valid"])
    return {
        "profile": profile_name,
        "token_count_stats": _token_count_stats(cases),
        "tokenization_summary": dict(aggregate["tokenization_summary"]),
        "interpretation": dict(interpretation),
        "full_clean": {
            "verdict": full_clean.get("verdict"),
            "case_count": full_clean.get("case_count"),
            "max_separation_layer": _layer_value(full_clean.get("max_separation_layer")),
            "first_positive_layer": _layer_value(full_clean.get("first_positive_layer")),
            "stable_birth_layer": _layer_value(full_clean.get("stable_birth_layer")),
        },
        "trimmed_clean": {
            "verdict": trimmed_clean.get("verdict"),
            "case_count": trimmed_clean.get("case_count"),
            "max_separation_layer": _layer_value(trimmed_clean.get("max_separation_layer")),
            "first_positive_layer": _layer_value(trimmed_clean.get("first_positive_layer")),
            "stable_birth_layer": _layer_value(trimmed_clean.get("stable_birth_layer")),
        },
        "full_all": {
            "verdict": full_all.get("verdict"),
            "case_count": full_all.get("case_count"),
            "max_separation_layer": _layer_value(full_all.get("max_separation_layer")),
            "first_positive_layer": _layer_value(full_all.get("first_positive_layer")),
            "stable_birth_layer": _layer_value(full_all.get("stable_birth_layer")),
        },
        "trimmed_all": {
            "verdict": trimmed_all.get("verdict"),
            "case_count": trimmed_all.get("case_count"),
            "max_separation_layer": _layer_value(trimmed_all.get("max_separation_layer")),
            "first_positive_layer": _layer_value(trimmed_all.get("first_positive_layer")),
            "stable_birth_layer": _layer_value(trimmed_all.get("stable_birth_layer")),
        },
        "trimmed_group_transitions": _group_transition_table(trimmed_clean),
        "full_group_transitions": _group_transition_table(full_clean),
        "aggregate": aggregate,
        "results": strip_mean_directions(results),
        "cases": [_case_to_dict(case) for case in cases],
    }


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    layers: list[int],
    profile_summaries: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen Anchor Geometry Profile Sweep",
        "",
        "## Summary",
        "",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        f"- Layers analyzed: `{layers[0]}..{layers[-1]}`",
        "",
        "## Crystallization summary by anchor span profile",
        "",
        "| profile | mean_words | support | trimmed verdict | first positive layer | stable birth layer | max separation layer | clean cases | noisy cases |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in profile_summaries:
        tokenization = dict(summary["tokenization_summary"])
        interpretation = dict(summary["interpretation"])
        trimmed_clean = dict(summary["trimmed_clean"])
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["profile"],
                    f"{float(summary['token_count_stats']['mean']):.3f}",
                    str(interpretation.get("support_after_tokenization_controls")),
                    str(trimmed_clean.get("verdict")),
                    str(trimmed_clean.get("first_positive_layer") if trimmed_clean.get("first_positive_layer") is not None else "n/a"),
                    str(trimmed_clean.get("stable_birth_layer") if trimmed_clean.get("stable_birth_layer") is not None else "n/a"),
                    str(trimmed_clean.get("max_separation_layer") if trimmed_clean.get("max_separation_layer") is not None else "n/a"),
                    str(tokenization.get("clean_case_count", "n/a")),
                    str(tokenization.get("noisy_case_count", "n/a")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Group transition summary — trimmed span / clean only",
            "",
            "| profile | group | class | transitional | sign_changes | first content-like layer | first procedure-like layer | strongest content-like layer | strongest procedure-like layer |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in profile_summaries:
        for group_name, transition in sorted(summary["trimmed_group_transitions"].items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        summary["profile"],
                        group_name,
                        str(transition.get("anchor_class")),
                        str(bool(transition.get("transitional", False))),
                        str(int(transition.get("sign_changes", 0) or 0)),
                        str(transition.get("first_content_like_layer") if transition.get("first_content_like_layer") is not None else "n/a"),
                        str(transition.get("first_procedure_like_layer") if transition.get("first_procedure_like_layer") is not None else "n/a"),
                        str(transition.get("strongest_content_like_layer") if transition.get("strongest_content_like_layer") is not None else "n/a"),
                        str(transition.get("strongest_procedure_like_layer") if transition.get("strongest_procedure_like_layer") is not None else "n/a"),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Reading guide",
            "",
            "- `first positive layer` — первый слой, где separation score уже устойчиво позитивен.",
            "- `stable birth layer` — первый слой, после которого несколько подряд слоёв держат separation, то есть место рождения кристаллизации.",
            "- `max separation layer` — где различие между content-like и procedure-like геометрией максимально.",
            "- `transitional = True` — знак polarity margin меняется по глубине, то есть группа реально переходит между режимами, а не сидит в одном режиме с начала до конца.",
            "",
            "Это geometry-only sweep: он намеренно не использует generation quality как основную цель и нужен именно для поиска траекторий, переходов и слоя рождения кристаллизации.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep anchor span profiles for geometry-only crystallization analysis.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=160)
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
        default=ROOT / "archive" / "qwen_anchor_geometry_profile_sweep.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_geometry_profile_sweep.md",
    )
    args = parser.parse_args()

    for profile in args.profiles:
        if profile not in list_anchor_span_profiles():
            raise ValueError(f"unknown profile: {profile}")

    device = torch.device(args.device)
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model,
        cfg=TOY_CONFIG,
        device=device,
        torch_dtype=torch.float16 if "cuda" in str(device) else None,
        low_cpu_mem_usage=True,
    )
    overlay.eval()

    layers = list_model_layers(int(getattr(overlay, "model_num_hidden_layers", 0)))
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
        results = [
            analyze_case_geometry(
                overlay=overlay,
                case=case,
                layers=layers,
                max_length=int(args.max_length),
                device=device,
            )
            for case in cases
        ]
        aggregate = aggregate_results(results, layers=layers)
        interpretation = infer_overall_interpretation(aggregate)
        summary = summarize_profile(
            profile_name=profile_name,
            cases=cases,
            results=results,
            aggregate=aggregate,
            interpretation=interpretation,
        )
        profile_summaries.append(summary)
        profile_payloads.append(
            {
                "profile": profile_name,
                "cases": summary["cases"],
                "results": summary["results"],
                "aggregate": aggregate,
                "interpretation": interpretation,
            }
        )

    profile_summaries.sort(key=lambda item: float(item["token_count_stats"]["mean"] or 0.0))
    best_geometry_profile = max(profile_summaries, key=_profile_rank_key)
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": args.model,
            "device": str(device),
            "max_length": int(args.max_length),
            "probe_layers": layers,
            "profiles": list(args.profiles),
        },
        "curve_points": [
            {
                "profile": summary["profile"],
                "mean_word_count": summary["token_count_stats"]["mean"],
                "support": dict(summary["interpretation"]).get("support_after_tokenization_controls"),
                "trimmed_verdict": dict(summary["trimmed_clean"]).get("verdict"),
                "trimmed_first_positive_layer": dict(summary["trimmed_clean"]).get("first_positive_layer"),
                "trimmed_stable_birth_layer": dict(summary["trimmed_clean"]).get("stable_birth_layer"),
                "trimmed_max_separation_layer": dict(summary["trimmed_clean"]).get("max_separation_layer"),
            }
            for summary in profile_summaries
        ],
        "best_geometry_profile": {
            key: value
            for key, value in best_geometry_profile.items()
            if key not in {"aggregate", "results", "cases"}
        },
        "profiles": profile_payloads,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=str(device),
        layers=layers,
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
