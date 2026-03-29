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


def build_auxiliary_revision_report_data(
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
        row = class_by_family.get(family, {"classification": "unclassified"})
        family_row = {
            "family": family,
            "classification": row["classification"],
            "stable_matches": int(stable.get("auxiliary_revision_matched_anchor_count", 0)),
            "conflict_matches": int(conflict.get("auxiliary_revision_matched_anchor_count", 0)),
            "match_gap": int(conflict.get("auxiliary_revision_matched_anchor_count", 0))
            - int(stable.get("auxiliary_revision_matched_anchor_count", 0)),
            "stable_revise_gain": int(stable.get("auxiliary_revision_revise_gain", 0)),
            "conflict_revise_gain": int(conflict.get("auxiliary_revision_revise_gain", 0)),
            "revise_gain_gap": int(conflict.get("auxiliary_revision_revise_gain", 0))
            - int(stable.get("auxiliary_revision_revise_gain", 0)),
            "stable_retire_delta": int(stable.get("auxiliary_revision_retire_delta", 0)),
            "conflict_retire_delta": int(conflict.get("auxiliary_revision_retire_delta", 0)),
            "retire_delta_gap": int(conflict.get("auxiliary_revision_retire_delta", 0))
            - int(stable.get("auxiliary_revision_retire_delta", 0)),
            "stable_alt_prob": float(stable.get("auxiliary_revision_mean_alt_prob", 0.0)),
            "conflict_alt_prob": float(conflict.get("auxiliary_revision_mean_alt_prob", 0.0)),
            "alt_prob_gap": float(conflict.get("auxiliary_revision_mean_alt_prob", 0.0))
            - float(stable.get("auxiliary_revision_mean_alt_prob", 0.0)),
            "match_win": int(conflict.get("auxiliary_revision_matched_anchor_count", 0))
            > int(stable.get("auxiliary_revision_matched_anchor_count", 0)),
            "revise_gain_win": int(conflict.get("auxiliary_revision_revise_gain", 0))
            > int(stable.get("auxiliary_revision_revise_gain", 0)),
            "stable_aux_texts": [item["proposal_text"] for item in stable.get("auxiliary_proposals", [])],
            "conflict_aux_texts": [item["proposal_text"] for item in conflict.get("auxiliary_proposals", [])],
        }
        families.append(family_row)

    summary = {
        "family_count": len(families),
        "match_wins": sum(1 for row in families if row["match_win"]),
        "revise_gain_wins": sum(1 for row in families if row["revise_gain_win"]),
        "mean_match_gap": (
            sum(row["match_gap"] for row in families) / len(families) if families else 0.0
        ),
        "mean_revise_gain_gap": (
            sum(row["revise_gain_gap"] for row in families) / len(families) if families else 0.0
        ),
        "mean_retire_delta_gap": (
            sum(row["retire_delta_gap"] for row in families) / len(families) if families else 0.0
        ),
        "future_rescue_count": sum(1 for row in families if row["classification"] == "future_rescue"),
        "future_rescue_revise_gain_wins": sum(
            1 for row in families if row["classification"] == "future_rescue" and row["revise_gain_win"]
        ),
    }
    return {"summary": summary, "families": families}


def build_markdown_report(report_data: dict[str, Any]) -> str:
    summary = report_data["summary"]
    families = report_data["families"]
    future_rescue_rows = [row for row in families if row["classification"] == "future_rescue"]

    lines = [
        "# Qwen Auxiliary Revision Report",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"- Families analyzed: `{summary['family_count']}`",
        f"- Conflict matched-anchor wins: `{summary['match_wins']}/{summary['family_count']}`",
        f"- Conflict revise-gain wins: `{summary['revise_gain_wins']}/{summary['family_count']}`",
        f"- Mean match gap (conflict - stable): `{summary['mean_match_gap']:.4f}`",
        f"- Mean revise-gain gap (conflict - stable): `{summary['mean_revise_gain_gap']:.4f}`",
        f"- Mean retire-delta gap (conflict - stable): `{summary['mean_retire_delta_gap']:.4f}`",
        f"- Future-rescue families: `{summary['future_rescue_count']}`",
        f"- Future-rescue revise-gain wins: `{summary['future_rescue_revise_gain_wins']}/{summary['future_rescue_count'] or 1}`",
        "",
        "## Family table",
        "",
        "| Family | Class | Stable matches | Conflict matches | Match gap | Stable revise gain | Conflict revise gain | Revise gap | Stable alt prob | Conflict alt prob |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in families:
        lines.append(
            f"| {row['family']} | {row['classification']} | {row['stable_matches']} | {row['conflict_matches']} | {row['match_gap']:+d} | "
            f"{row['stable_revise_gain']:+d} | {row['conflict_revise_gain']:+d} | {row['revise_gain_gap']:+d} | "
            f"{row['stable_alt_prob']:.4f} | {row['conflict_alt_prob']:.4f} |"
        )

    lines.extend(["", "## Future-rescue highlights", ""])
    if not future_rescue_rows:
        lines.append("- No `future_rescue` families in the current analysis.")
    else:
        for row in future_rescue_rows:
            lines.extend(
                [
                    f"### {row['family']}",
                    f"- match gap: `{row['match_gap']:+d}`",
                    f"- revise-gain gap: `{row['revise_gain_gap']:+d}`",
                    f"- stable auxiliary spans: `{'; '.join(row['stable_aux_texts']) if row['stable_aux_texts'] else 'none'}`",
                    f"- conflict auxiliary spans: `{'; '.join(row['conflict_aux_texts']) if row['conflict_aux_texts'] else 'none'}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This report asks whether auxiliary future-hint proposals change revision behaviour, not just whether they exist.",
            "- A positive revise-gain on conflict cases would mean the proposal-like hints are starting to push the controller toward `revise` rather than the base path.",
            "- If rescue families show stronger matches but little revise gain, the next bottleneck is probably arbiter calibration rather than hint extraction.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate auxiliary revision effects from Qwen future-influence diagnostics.")
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
        default=ROOT / "docs" / "research" / "qwen_auxiliary_revision_report.md",
    )
    args = parser.parse_args()

    report_data = build_auxiliary_revision_report_data(
        anchor_payload=load_payload(args.anchor_json),
        future_payload=load_payload(args.future_json),
    )
    report = build_markdown_report(report_data)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
