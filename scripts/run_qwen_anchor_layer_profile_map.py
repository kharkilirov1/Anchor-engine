from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import re
from pathlib import Path
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
from scripts.run_qwen_anchor_geometry_probe import analyze_case_geometry


PLOT_METRICS = (
    "rank1_explained_variance",
    "adjacent_cosine_coherence",
    "path_tortuosity",
)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "item"


def _mode_series(mode_payload: dict[str, Any]) -> list[dict[str, Any]]:
    if dict(mode_payload).get("status") != "ok":
        return []
    series: list[dict[str, Any]] = []
    for layer_result in mode_payload.get("layer_results", []):
        metrics = dict(layer_result.get("metrics", {}))
        series.append(
            {
                "layer": int(layer_result["layer"]),
                "rank1_explained_variance": metrics.get("rank1_explained_variance"),
                "adjacent_cosine_coherence": metrics.get("adjacent_cosine_coherence"),
                "path_tortuosity": metrics.get("path_tortuosity"),
            }
        )
    return series


def _peak_layer(series: list[dict[str, Any]], metric_name: str, *, maximize: bool) -> int | None:
    valid = [item for item in series if item.get(metric_name) is not None]
    if not valid:
        return None
    key = lambda item: float(item[metric_name])
    best = max(valid, key=key) if maximize else min(valid, key=key)
    return int(best["layer"])


def summarize_case_profile(result: dict[str, Any]) -> dict[str, Any]:
    modes = dict(result.get("modes", {}))
    summary: dict[str, Any] = {
        "name": result.get("name"),
        "anchor_class": result.get("anchor_class"),
        "anchor_group": result.get("anchor_group"),
        "status": result.get("status"),
        "mode_summaries": {},
    }
    if result.get("status") != "ok":
        summary["skip_reason"] = result.get("skip_reason")
        return summary
    for mode_name in ("full_span", "trimmed_span"):
        mode_payload = dict(modes.get(mode_name, {}))
        series = _mode_series(mode_payload)
        summary["mode_summaries"][mode_name] = {
            "token_count": dict(mode_payload.get("span", {})).get("token_count"),
            "rank1_peak_layer": _peak_layer(series, "rank1_explained_variance", maximize=True),
            "coherence_peak_layer": _peak_layer(series, "adjacent_cosine_coherence", maximize=True),
            "tortuosity_min_layer": _peak_layer(series, "path_tortuosity", maximize=False),
            "series": series,
        }
    return summary


def plot_case_profile(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = summarize_case_profile(result)
    if result.get("status") != "ok":
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)
    mode_order = ("full_span", "trimmed_span")
    labels = {
        "rank1_explained_variance": "r1 EV",
        "adjacent_cosine_coherence": "coherence",
        "path_tortuosity": "tortuosity",
    }
    colors = {
        "rank1_explained_variance": "#1f77b4",
        "adjacent_cosine_coherence": "#2ca02c",
        "path_tortuosity": "#d62728",
    }
    for ax, mode_name in zip(axes, mode_order):
        mode_summary = dict(summary["mode_summaries"].get(mode_name, {}))
        series = list(mode_summary.get("series", []))
        if not series:
            ax.set_title(f"{mode_name}: unavailable")
            ax.axis("off")
            continue
        layers = [int(item["layer"]) for item in series]
        for metric_name in PLOT_METRICS:
            values = [item.get(metric_name) for item in series]
            ax.plot(layers, values, marker="o", linewidth=1.8, markersize=3.5, label=labels[metric_name], color=colors[metric_name])
        rank1_peak = mode_summary.get("rank1_peak_layer")
        if rank1_peak is not None:
            ax.axvline(int(rank1_peak), color=colors["rank1_explained_variance"], linestyle="--", alpha=0.35)
        coherence_peak = mode_summary.get("coherence_peak_layer")
        if coherence_peak is not None:
            ax.axvline(int(coherence_peak), color=colors["adjacent_cosine_coherence"], linestyle=":", alpha=0.35)
        tort_min = mode_summary.get("tortuosity_min_layer")
        if tort_min is not None:
            ax.axvline(int(tort_min), color=colors["path_tortuosity"], linestyle="-.", alpha=0.35)
        ax.set_title(f"{mode_name} | tokens={mode_summary.get('token_count', 'n/a')}")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("metric value")
    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        f"{result['name']} | {result['anchor_group']} | {result['anchor_class']}",
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    profiles: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen Anchor Layer Profile Map",
        "",
        "## Summary",
        "",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "",
        "Этот отчёт не оптимизирует inference quality. Он показывает per-case layer profiles, чтобы глазами увидеть, где растёт `r1`, где падает `tortuosity`, где появляется `coherence`, и как это меняется от длины anchor.",
        "",
    ]
    for profile_entry in profiles:
        lines.extend(
            [
                f"## Profile: `{profile_entry['profile']}`",
                "",
                "| case | class | full r1 peak | full coherence peak | full tortuosity min | trimmed r1 peak | trimmed coherence peak | trimmed tortuosity min | figure |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for case_summary in profile_entry["case_summaries"]:
            if case_summary.get("status") != "ok":
                lines.append(
                    f"| {case_summary['name']} | {case_summary['anchor_class']} | n/a | n/a | n/a | n/a | n/a | n/a | skipped |"
                )
                continue
            full = dict(case_summary["mode_summaries"]["full_span"])
            trimmed = dict(case_summary["mode_summaries"]["trimmed_span"])
            fig_rel = case_summary.get("figure_relpath")
            fig_md = f"[plot]({fig_rel})" if fig_rel else "n/a"
            lines.append(
                "| "
                + " | ".join(
                    [
                        case_summary["name"],
                        case_summary["anchor_class"],
                        str(full.get("rank1_peak_layer") if full.get("rank1_peak_layer") is not None else "n/a"),
                        str(full.get("coherence_peak_layer") if full.get("coherence_peak_layer") is not None else "n/a"),
                        str(full.get("tortuosity_min_layer") if full.get("tortuosity_min_layer") is not None else "n/a"),
                        str(trimmed.get("rank1_peak_layer") if trimmed.get("rank1_peak_layer") is not None else "n/a"),
                        str(trimmed.get("coherence_peak_layer") if trimmed.get("coherence_peak_layer") is not None else "n/a"),
                        str(trimmed.get("tortuosity_min_layer") if trimmed.get("tortuosity_min_layer") is not None else "n/a"),
                        fig_md,
                    ]
                )
                + " |"
            )
        lines.append("")
        for case_summary in profile_entry["case_summaries"]:
            if case_summary.get("status") != "ok":
                continue
            lines.append(f"### {case_summary['name']}")
            lines.append("")
            if case_summary.get("figure_relpath"):
                lines.append(f"![{case_summary['name']}]({case_summary['figure_relpath']})")
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-case layer profile maps for Qwen anchor geometry.")
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
        default=ROOT / "archive" / "qwen_anchor_layer_profile_map.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_layer_profile_map.md",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=ROOT / "docs" / "research" / "figures" / "qwen_anchor_layer_profiles",
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
    for profile_name in args.profiles:
        cases = make_qwen_anchor_geometry_cases(anchor_span_profile=profile_name)
        if args.case_name:
            cases = [case for case in cases if case.name == args.case_name]
        if args.limit is not None:
            cases = cases[: max(int(args.limit), 0)]
        if not cases:
            raise ValueError(f"no cases selected for profile {profile_name}")
        raw_results = [
            analyze_case_geometry(
                overlay=overlay,
                case=case,
                layers=layers,
                max_length=int(args.max_length),
                device=device,
            )
            for case in cases
        ]
        case_summaries: list[dict[str, Any]] = []
        for result in raw_results:
            case_summary = summarize_case_profile(result)
            if result.get("status") == "ok":
                figure_name = f"{_slug(profile_name)}_{_slug(str(result['name']))}.png"
                figure_path = args.figure_dir / figure_name
                plot_case_profile(result, figure_path)
                case_summary["figure_path"] = str(figure_path)
                case_summary["figure_relpath"] = Path(
                    Path("figures") / "qwen_anchor_layer_profiles" / figure_name
                ).as_posix()
            case_summaries.append(case_summary)
        profile_payloads.append(
            {
                "profile": profile_name,
                "cases": [asdict(case) for case in cases],
                "results": raw_results,
                "case_summaries": case_summaries,
            }
        )

    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": args.model,
            "device": str(device),
            "max_length": int(args.max_length),
            "probe_layers": layers,
            "profiles": list(args.profiles),
        },
        "profiles": profile_payloads,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=str(device),
        profiles=profile_payloads,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")
    print(f"saved_figures={args.figure_dir}")


if __name__ == "__main__":
    main()
