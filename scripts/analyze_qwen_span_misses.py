from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_by_family(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for item in results:
        paired.setdefault(item["family"], {})[item["expected_mode"]] = item
    return paired


def classify_family(
    delta_joint: bool,
    anchor_future_win: bool,
    overlap_win: bool,
) -> str:
    if delta_joint and anchor_future_win and overlap_win:
        return "aligned"
    if not delta_joint and anchor_future_win:
        return "future_rescue"
    if delta_joint and not anchor_future_win:
        return "delta_only"
    return "both_weak"


def build_analysis(
    anchor_payload: dict[str, Any],
    future_payload: dict[str, Any],
) -> dict[str, Any]:
    anchor_pairs = pair_by_family(anchor_payload["results"])
    future_pairs = pair_by_family(future_payload["results"])
    families = sorted(set(anchor_pairs) & set(future_pairs))

    rows: list[dict[str, Any]] = []
    for family in families:
        stable_anchor = anchor_pairs[family]["stable"]
        conflict_anchor = anchor_pairs[family]["conflict"]
        stable_future = future_pairs[family]["stable"]
        conflict_future = future_pairs[family]["conflict"]

        pressure_gap = (
            conflict_anchor["mean_contradiction_pressure"]
            - stable_anchor["mean_contradiction_pressure"]
        )
        viability_gap = (
            conflict_anchor["mean_viability"] - stable_anchor["mean_viability"]
        )
        anchor_future_gap = (
            conflict_future["anchor_position_mean_future_influence"]
            - stable_future["anchor_position_mean_future_influence"]
        )
        overlap_gap = (
            conflict_future.get("future_span_overlap_ratio", 0.0)
            - stable_future.get("future_span_overlap_ratio", 0.0)
        )
        delta_joint = pressure_gap > 0.0 and viability_gap < 0.0
        anchor_future_win = anchor_future_gap > 0.0
        overlap_win = overlap_gap > 0.0
        rows.append(
            {
                "family": family,
                "pressure_gap": pressure_gap,
                "viability_gap": viability_gap,
                "anchor_future_gap": anchor_future_gap,
                "overlap_gap": overlap_gap,
                "delta_joint": delta_joint,
                "anchor_future_win": anchor_future_win,
                "overlap_win": overlap_win,
                "classification": classify_family(
                    delta_joint=delta_joint,
                    anchor_future_win=anchor_future_win,
                    overlap_win=overlap_win,
                ),
            }
        )

    summary = {
        "family_count": len(rows),
        "aligned_count": sum(1 for row in rows if row["classification"] == "aligned"),
        "future_rescue_count": sum(1 for row in rows if row["classification"] == "future_rescue"),
        "delta_only_count": sum(1 for row in rows if row["classification"] == "delta_only"),
        "both_weak_count": sum(1 for row in rows if row["classification"] == "both_weak"),
    }
    return {"families": rows, "summary": summary}


def build_markdown_report(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    rows = analysis["families"]
    lines = [
        "# Qwen Span Miss Analysis",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"- Families analyzed: `{summary['family_count']}`",
        f"- Fully aligned families: `{summary['aligned_count']}`",
        f"- Future-rescue families: `{summary['future_rescue_count']}`",
        f"- Delta-only families: `{summary['delta_only_count']}`",
        f"- Both-weak families: `{summary['both_weak_count']}`",
        "",
        "## Family table",
        "",
        "| Family | Pressure gap | Viability gap | Anchor-future gap | Span-overlap gap | Classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['pressure_gap']:.4f} | {row['viability_gap']:.4f} | "
            f"{row['anchor_future_gap']:.4f} | {row['overlap_gap']:.4f} | {row['classification']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `aligned` means delta diagnostics and future-attribution spans both move in the expected conflict direction.",
            "- `future_rescue` means the current detector misses the family, but anchor-position future influence still rises on the conflict case.",
            "- `delta_only` means the existing detector works better than the future-attribution overlay on that family.",
            "- `both_weak` means neither signal is currently convincing enough.",
            "",
            "## Current reading",
            "",
            "- The most interesting families are `future_rescue`, because they are candidates where future-conditioned attribution may expose missed anchor spans.",
            "- The most important near-term target is `both_weak`, because those families likely need better prompts, better thresholds, or a stronger span aggregation rule.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze where future-influence spans rescue or miss current Qwen anchor diagnostics.")
    parser.add_argument(
        "--anchor_json",
        type=Path,
        default=ROOT / "archive" / "qwen_probe_results.json",
    )
    parser.add_argument(
        "--future_json",
        type=Path,
        default=ROOT / "archive" / "qwen_future_influence_probe.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_span_miss_analysis.md",
    )
    args = parser.parse_args()

    analysis = build_analysis(
        anchor_payload=load_payload(args.anchor_json),
        future_payload=load_payload(args.future_json),
    )
    report = build_markdown_report(analysis)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
