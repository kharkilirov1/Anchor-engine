from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_history(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"invalid history payload: {path}")
    return payload


def _last(history: list[dict], key: str) -> float | None:
    if key not in history[-1]:
        return None
    return float(history[-1][key])


def _delta(history: list[dict], key: str) -> float | None:
    if key not in history[0] or key not in history[-1]:
        return None
    return float(history[-1][key]) - float(history[0][key])


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _judgement(baseline: list[dict], anchor: list[dict]) -> str:
    baseline_val = _last(baseline, "val_bpb")
    anchor_val = _last(anchor, "val_bpb")
    anchor_cpress = _delta(anchor, "anchor_contradiction")
    anchor_viable = _delta(anchor, "anchor_viability")
    anchor_prop = _last(anchor, "proposal_influence")

    lines: list[str] = []
    if baseline_val is not None and anchor_val is not None:
        if anchor_val < baseline_val:
            lines.append("Anchor stage finished with lower validation BPB than baseline.")
        elif anchor_val > baseline_val:
            lines.append("Baseline finished with lower validation BPB than anchor.")
        else:
            lines.append("Baseline and anchor finished with the same validation BPB.")

    if anchor_cpress is not None and anchor_viable is not None:
        if anchor_cpress < 0 and anchor_viable > 0:
            lines.append("Anchor health improved during the anchor run: contradiction went down while viability went up.")
        elif anchor_cpress < 0:
            lines.append("Anchor contradiction decreased during the anchor run.")
        elif anchor_viable > 0:
            lines.append("Anchor viability improved during the anchor run.")
        else:
            lines.append("Anchor health did not show a clear positive shift during the anchor run.")

    if anchor_prop is not None:
        if anchor_prop > 0:
            lines.append("Proposal path became active by the end of the anchor run.")
        else:
            lines.append("Proposal path remained inactive at the end of the anchor run.")

    return " ".join(lines)


def generate_compare_report(
    baseline_history_path: Path,
    anchor_history_path: Path,
    output_path: Path | None = None,
) -> Path:
    baseline = _load_history(baseline_history_path)
    anchor = _load_history(anchor_history_path)

    if output_path is None:
        output_path = ROOT / "docs" / "research" / "anchor_training_compare_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = [
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

    lines: list[str] = []
    lines.append("# Anchor vs Baseline Training Compare")
    lines.append("")
    lines.append(f"Date: {datetime.now().date().isoformat()}")
    lines.append(f"Baseline history: `{baseline_history_path}`")
    lines.append(f"Anchor history: `{anchor_history_path}`")
    lines.append("")
    lines.append("## Final metrics")
    lines.append("")
    lines.append("| Metric | Baseline final | Anchor final | Δ (Anchor - Baseline) |")
    lines.append("|---|---:|---:|---:|")
    for metric, _mode in metric_rows:
        b = _last(baseline, metric)
        a = _last(anchor, metric)
        if b is None and a is None:
            continue
        diff = None if b is None or a is None else a - b
        lines.append(f"| {metric} | {_fmt(b)} | {_fmt(a)} | {_fmt(diff)} |")

    lines.append("")
    lines.append("## In-run deltas")
    lines.append("")
    lines.append("| Metric | Baseline Δ | Anchor Δ |")
    lines.append("|---|---:|---:|")
    for metric, _mode in metric_rows:
        bd = _delta(baseline, metric)
        ad = _delta(anchor, metric)
        if bd is None and ad is None:
            continue
        lines.append(f"| {metric} | {_fmt(bd)} | {_fmt(ad)} |")

    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(_judgement(baseline, anchor))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate baseline-vs-anchor markdown compare report")
    parser.add_argument("baseline_history_path")
    parser.add_argument("anchor_history_path")
    parser.add_argument("--output", dest="output_path", default=None)
    args = parser.parse_args()

    report_path = generate_compare_report(
        baseline_history_path=Path(args.baseline_history_path),
        anchor_history_path=Path(args.anchor_history_path),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(report_path)
