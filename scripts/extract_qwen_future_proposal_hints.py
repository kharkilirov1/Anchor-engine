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

from scripts.analyze_qwen_span_misses import build_analysis


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def spans_overlap(span_a: dict[str, Any], span_b: dict[str, Any]) -> bool:
    return not (int(span_a["end"]) < int(span_b["start"]) or int(span_b["end"]) < int(span_a["start"]))


def extract_hint_candidates(
    future_payload: dict[str, Any],
    analysis: dict[str, Any],
    allowed_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = allowed_classes or {"future_rescue", "aligned"}
    class_by_family = {row["family"]: row for row in analysis["families"]}
    hints: list[dict[str, Any]] = []

    for item in future_payload["results"]:
        if item["expected_mode"] != "conflict":
            continue
        family_row = class_by_family.get(item["family"])
        if family_row is None or family_row["classification"] not in allowed:
            continue

        active_anchor_spans = item.get("active_anchor_spans", [])
        for span in item.get("future_spans", []):
            if any(spans_overlap(span, anchor_span) for anchor_span in active_anchor_spans):
                continue
            hint_score = float(span["mean_score"]) * (1.0 + max(0.0, float(family_row["anchor_future_gap"])))
            hints.append(
                {
                    "family": item["family"],
                    "case": item["name"],
                    "classification": family_row["classification"],
                    "hint_score": hint_score,
                    "span_start": int(span["start"]),
                    "span_end": int(span["end"]),
                    "span_text": span["text"],
                    "span_mean_score": float(span["mean_score"]),
                    "span_max_score": float(span["max_score"]),
                    "active_anchor_spans": active_anchor_spans,
                }
            )

    hints.sort(
        key=lambda item: (
            item["classification"] != "future_rescue",
            -item["hint_score"],
            -item["span_mean_score"],
        )
    )
    return hints


def build_markdown_report(
    analysis: dict[str, Any],
    hints: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen Future Proposal Hints",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"- Hint candidates: `{len(hints)}`",
        f"- Source families: `{len({item['family'] for item in hints})}`",
        "",
        "## Candidate spans",
        "",
        "| Family | Case | Class | Hint score | Span | Mean | Max | Text |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    for item in hints:
        lines.append(
            f"| {item['family']} | {item['case']} | {item['classification']} | {item['hint_score']:.4f} | "
            f"{item['span_start']}-{item['span_end']} | {item['span_mean_score']:.4f} | {item['span_max_score']:.4f} | `{item['span_text']}` |"
        )

    rescue_families = [row["family"] for row in analysis["families"] if row["classification"] == "future_rescue"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These are conflict-case future-influence spans that do not overlap the current active anchors.",
            "- `future_rescue` families are the most interesting, because they are cases where future-conditioned attribution may expose missed anchor spans.",
            f"- Current future-rescue families: `{', '.join(rescue_families) if rescue_families else 'none'}`",
            "- The next practical use for these spans is as experimental proposal hints or auxiliary anchor candidates, not as automatic replacements for the current detector.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract future-influence span hints for possible proposal candidates.")
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
        default=ROOT / "docs" / "research" / "qwen_future_proposal_hints.md",
    )
    args = parser.parse_args()

    anchor_payload = load_payload(args.anchor_json)
    future_payload = load_payload(args.future_json)
    analysis = build_analysis(anchor_payload, future_payload)
    hints = extract_hint_candidates(future_payload, analysis)
    report = build_markdown_report(analysis, hints)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
