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

from scripts.analyze_qwen_span_misses import build_analysis, pair_by_family


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_auxiliary_report_data(
    anchor_payload: dict[str, Any],
    future_payload: dict[str, Any],
) -> dict[str, Any]:
    analysis = build_analysis(anchor_payload=anchor_payload, future_payload=future_payload)
    future_pairs = pair_by_family(future_payload["results"])
    class_by_family = {row["family"]: row for row in analysis["families"]}

    families: list[dict[str, Any]] = []
    for family, pair in sorted(future_pairs.items()):
        if "stable" not in pair or "conflict" not in pair:
            continue
        stable = pair["stable"]
        conflict = pair["conflict"]
        row = class_by_family.get(family, {"classification": "unclassified", "anchor_future_gap": 0.0})
        stable_count = int(stable.get("auxiliary_proposal_count", 0))
        conflict_count = int(conflict.get("auxiliary_proposal_count", 0))
        stable_score = float(stable.get("auxiliary_mean_proposal_score", 0.0))
        conflict_score = float(conflict.get("auxiliary_mean_proposal_score", 0.0))
        family_row = {
            "family": family,
            "classification": row["classification"],
            "anchor_future_gap": float(row.get("anchor_future_gap", 0.0)),
            "stable_aux_count": stable_count,
            "conflict_aux_count": conflict_count,
            "aux_count_gap": conflict_count - stable_count,
            "stable_aux_score": stable_score,
            "conflict_aux_score": conflict_score,
            "aux_score_gap": conflict_score - stable_score,
            "count_win": conflict_count > stable_count,
            "score_win": conflict_score > stable_score,
            "stable_aux_texts": [item["proposal_text"] for item in stable.get("auxiliary_proposals", [])],
            "conflict_aux_texts": [item["proposal_text"] for item in conflict.get("auxiliary_proposals", [])],
        }
        families.append(family_row)

    summary = {
        "family_count": len(families),
        "count_wins": sum(1 for row in families if row["count_win"]),
        "score_wins": sum(1 for row in families if row["score_win"]),
        "joint_wins": sum(1 for row in families if row["count_win"] and row["score_win"]),
        "mean_count_gap": (
            sum(row["aux_count_gap"] for row in families) / len(families) if families else 0.0
        ),
        "mean_score_gap": (
            sum(row["aux_score_gap"] for row in families) / len(families) if families else 0.0
        ),
        "future_rescue_count": sum(1 for row in families if row["classification"] == "future_rescue"),
        "future_rescue_joint_wins": sum(
            1 for row in families if row["classification"] == "future_rescue" and row["count_win"] and row["score_win"]
        ),
    }
    return {"summary": summary, "families": families, "analysis": analysis}


def build_markdown_report(report_data: dict[str, Any]) -> str:
    summary = report_data["summary"]
    families = report_data["families"]
    future_rescue_rows = [row for row in families if row["classification"] == "future_rescue"]

    lines = [
        "# Qwen Auxiliary Proposal Report",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"- Families analyzed: `{summary['family_count']}`",
        f"- Conflict proposal-count wins: `{summary['count_wins']}/{summary['family_count']}`",
        f"- Conflict proposal-score wins: `{summary['score_wins']}/{summary['family_count']}`",
        f"- Joint wins (count + score): `{summary['joint_wins']}/{summary['family_count']}`",
        f"- Mean proposal-count gap (conflict - stable): `{summary['mean_count_gap']:.4f}`",
        f"- Mean proposal-score gap (conflict - stable): `{summary['mean_score_gap']:.4f}`",
        f"- Future-rescue families: `{summary['future_rescue_count']}`",
        f"- Future-rescue joint wins: `{summary['future_rescue_joint_wins']}/{summary['future_rescue_count'] or 1}`",
        "",
        "## Family table",
        "",
        "| Family | Class | Stable count | Conflict count | Count gap | Stable score | Conflict score | Score gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in families:
        lines.append(
            f"| {row['family']} | {row['classification']} | {row['stable_aux_count']} | {row['conflict_aux_count']} | "
            f"{row['aux_count_gap']:+d} | {row['stable_aux_score']:.4f} | {row['conflict_aux_score']:.4f} | {row['aux_score_gap']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Future-rescue highlights",
            "",
        ]
    )

    if not future_rescue_rows:
        lines.append("- No `future_rescue` families in the current analysis.")
    else:
        for row in future_rescue_rows:
            lines.extend(
                [
                    f"### {row['family']}",
                    f"- count gap: `{row['aux_count_gap']:+d}`",
                    f"- score gap: `{row['aux_score_gap']:+.4f}`",
                    f"- stable auxiliary spans: `{'; '.join(row['stable_aux_texts']) if row['stable_aux_texts'] else 'none'}`",
                    f"- conflict auxiliary spans: `{'; '.join(row['conflict_aux_texts']) if row['conflict_aux_texts'] else 'none'}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This report evaluates the new auxiliary proposal-like spans built from high future-influence regions.",
            "- A useful result is not absolute proposal volume, but whether conflict prompts produce more or stronger auxiliary proposals than their stable controls.",
            "- `future_rescue` families are the most important target, because they are where future-conditioned attribution may compensate for detector misses.",
            "- These auxiliary proposals are still offline diagnostics; they are not yet connected to decoding or revision control.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate auxiliary future-hint proposals on the Qwen probe suite.")
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
        default=ROOT / "docs" / "research" / "qwen_auxiliary_proposal_report.md",
    )
    args = parser.parse_args()

    report_data = build_auxiliary_report_data(
        anchor_payload=load_payload(args.anchor_json),
        future_payload=load_payload(args.future_json),
    )
    report = build_markdown_report(report_data)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
