from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_history(history_path: Path) -> list[dict]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("training history must be a JSON list")
    if not payload:
        raise ValueError("training history is empty")
    return payload


def _metric_delta(history: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in history if key in row]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def _metric_best(history: list[dict], key: str, mode: str) -> float | None:
    values = [float(row[key]) for row in history if key in row]
    if not values:
        return None
    if mode == "min":
        return min(values)
    return max(values)


def generate_training_report(
    history_path: Path,
    output_path: Path | None = None,
) -> Path:
    history = _load_history(history_path)

    if output_path is None:
        output_path = ROOT / "docs" / "research" / "anchor_training_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracked_metrics = [
        ("loss", "min"),
        ("ce_loss", "min"),
        ("bpb", "min"),
        ("val_bpb", "min"),
        ("anchors_active", "max"),
        ("anchor_contradiction", "min"),
        ("anchor_viability", "max"),
        ("anchor_dead_end", "min"),
        ("proposal_influence", "max"),
        ("proposal_blend", "max"),
        ("strong_retire_gap", "min"),
        ("detector_alignment_loss", "min"),
        ("context_stability_loss", "min"),
    ]

    start = history[0]
    end = history[-1]
    lines: list[str] = []
    lines.append("# Anchor Training Report")
    lines.append("")
    lines.append(f"Date: {datetime.now().date().isoformat()}")
    lines.append(f"Source history: `{history_path}`")
    lines.append(f"Steps captured: {len(history)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("This report summarizes the current training dynamics captured from `train.py --stage anchor --history_path ...`.")
    lines.append("It is intended as the training-side analogue of the semantic probe report.")
    lines.append("")
    lines.append("## Metric deltas")
    lines.append("")
    lines.append("| Metric | Start | End | Delta | Best |")
    lines.append("|---|---:|---:|---:|---:|")
    for metric, mode in tracked_metrics:
        if metric not in start or metric not in end:
            continue
        delta = _metric_delta(history, metric)
        best = _metric_best(history, metric, mode)
        if delta is None or best is None:
            continue
        lines.append(
            f"| {metric} | {float(start[metric]):.4f} | {float(end[metric]):.4f} | {delta:.4f} | {best:.4f} |"
        )
    lines.append("")
    lines.append("## Step table")
    lines.append("")
    present_columns = [
        "step",
        "loss",
        "ce_loss",
        "bpb",
        "val_bpb",
        "anchors_active",
        "anchor_contradiction",
        "anchor_viability",
        "anchor_dead_end",
        "proposal_influence",
        "proposal_blend",
        "strong_retire_gap",
        "detector_alignment_loss",
        "context_stability_loss",
    ]
    columns = [col for col in present_columns if any(col in row for row in history)]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---:|" * len(columns))
    for row in history:
        rendered: list[str] = []
        for col in columns:
            value = row.get(col)
            if value is None:
                rendered.append("")
            else:
                rendered.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(rendered) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `anchor_contradiction` should ideally decrease or remain controlled as training progresses.")
    lines.append("- `anchor_viability` should not collapse while CE/BPB improves.")
    lines.append("- `proposal_influence` and `proposal_blend` should stay observable rather than collapsing to zero if alternative-hypothesis machinery remains active.")
    lines.append("- `detector_alignment_loss` and `context_stability_loss` are stabilizers, not end goals; they should help without dominating CE optimization.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate markdown report from Anchor V1 training history")
    parser.add_argument("history_path")
    parser.add_argument("--output", dest="output_path", default=None)
    args = parser.parse_args()

    report_path = generate_training_report(
        history_path=Path(args.history_path),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(report_path)
