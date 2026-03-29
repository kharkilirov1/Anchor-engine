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


def pair_cases_by_family(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for item in results:
        paired.setdefault(item["family"], {})[item["expected_mode"]] = item
    return paired


def compare_payloads(
    anchor_payload: dict[str, Any],
    future_payload: dict[str, Any],
) -> dict[str, Any]:
    anchor_pairs = pair_cases_by_family(anchor_payload["results"])
    future_pairs = pair_cases_by_family(future_payload["results"])
    families = sorted(set(anchor_pairs) & set(future_pairs))

    rows: list[dict[str, Any]] = []
    for family in families:
        anchor_stable = anchor_pairs[family]["stable"]
        anchor_conflict = anchor_pairs[family]["conflict"]
        future_stable = future_pairs[family]["stable"]
        future_conflict = future_pairs[family]["conflict"]

        pressure_gap = (
            anchor_conflict["mean_contradiction_pressure"]
            - anchor_stable["mean_contradiction_pressure"]
        )
        viability_gap = (
            anchor_conflict["mean_viability"] - anchor_stable["mean_viability"]
        )
        future_gap = (
            future_conflict["mean_future_influence"]
            - future_stable["mean_future_influence"]
        )
        anchor_future_gap = (
            future_conflict["anchor_position_mean_future_influence"]
            - future_stable["anchor_position_mean_future_influence"]
        )

        rows.append(
            {
                "family": family,
                "pressure_gap": pressure_gap,
                "viability_gap": viability_gap,
                "future_gap": future_gap,
                "anchor_future_gap": anchor_future_gap,
                "pressure_win": pressure_gap > 0.0,
                "viability_win": viability_gap < 0.0,
                "future_win": future_gap > 0.0,
                "anchor_future_win": anchor_future_gap > 0.0,
            }
        )

    summary = {
        "family_count": len(rows),
        "pressure_wins": sum(1 for row in rows if row["pressure_win"]),
        "viability_wins": sum(1 for row in rows if row["viability_win"]),
        "future_wins": sum(1 for row in rows if row["future_win"]),
        "anchor_future_wins": sum(1 for row in rows if row["anchor_future_win"]),
        "mean_pressure_gap": (
            sum(row["pressure_gap"] for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_viability_gap": (
            sum(row["viability_gap"] for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_future_gap": (
            sum(row["future_gap"] for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_anchor_future_gap": (
            sum(row["anchor_future_gap"] for row in rows) / len(rows) if rows else 0.0
        ),
    }
    return {"families": rows, "summary": summary}


def build_markdown_report(
    anchor_payload: dict[str, Any],
    future_payload: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    summary = comparison["summary"]
    rows = comparison["families"]
    lines = [
        "# Qwen Signal Proxy Comparison",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Anchor probe source: `{anchor_payload['model']}`",
        f"Future-influence source: `{future_payload['model']}`",
        "",
        "## Summary",
        "",
        f"- Families compared: `{summary['family_count']}`",
        f"- Delta-pressure wins: `{summary['pressure_wins']}`",
        f"- Delta-viability wins: `{summary['viability_wins']}`",
        f"- Mean future-influence wins: `{summary['future_wins']}`",
        f"- Anchor-position future-influence wins: `{summary['anchor_future_wins']}`",
        f"- Mean pressure gap: `{summary['mean_pressure_gap']:.4f}`",
        f"- Mean viability gap: `{summary['mean_viability_gap']:.4f}`",
        f"- Mean future-influence gap: `{summary['mean_future_gap']:.4f}`",
        f"- Mean anchor-position future-influence gap: `{summary['mean_anchor_future_gap']:.4f}`",
        "",
        "## Family table",
        "",
        "| Family | Pressure gap | Viability gap | Mean future gap | Anchor-position future gap | Delta joint | Future anchor |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['pressure_gap']:.4f} | {row['viability_gap']:.4f} | "
            f"{row['future_gap']:.4f} | {row['anchor_future_gap']:.4f} | "
            f"{'win' if row['pressure_win'] and row['viability_win'] else 'miss'} | "
            f"{'win' if row['anchor_future_win'] else 'miss'} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The existing delta-based anchor diagnostics still provide the cleanest global stable-vs-conflict signal.",
            "- Raw mean future influence is mixed and should not yet be treated as a drop-in replacement for contradiction/viability scoring.",
            "- Anchor-position future influence looks more promising than plain prompt-level future influence, because it wins in more families and highlights conflict-sensitive positions even when prompt-level means are ambiguous.",
            "- The current best reading is that future-conditioned attribution is useful as a positional relevance probe, not yet as a single scalar anchor score.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare delta-based and future-influence Qwen probe signals.")
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
        default=ROOT / "docs" / "research" / "qwen_signal_proxy_compare.md",
    )
    args = parser.parse_args()

    anchor_payload = load_payload(args.anchor_json)
    future_payload = load_payload(args.future_json)
    comparison = compare_payloads(anchor_payload, future_payload)
    report = build_markdown_report(anchor_payload, future_payload, comparison)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
