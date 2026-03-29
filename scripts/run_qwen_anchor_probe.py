from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, UTC
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_probe_cases import make_qwen_probe_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay


def collect_case_result(
    overlay: QwenAnchorOverlay,
    case_name: str,
    case_description: str,
    case_prompt: str,
    expected_mode: str,
    max_length: int,
) -> dict[str, Any]:
    out, batch = overlay.analyze_texts([case_prompt], max_length=max_length)
    diag = out["anchor_diagnostics"]
    proposal_diag = out["proposal_diagnostics"]
    return {
        "name": case_name,
        "description": case_description,
        "expected_mode": expected_mode,
        "tokens": int(batch["input_ids"].numel()),
        "num_active": int(diag["num_active"]),
        "mean_contradiction_pressure": float(diag["mean_contradiction_pressure"]),
        "mean_viability": float(diag["mean_viability"]),
        "dead_end_count": int(diag["dead_end_count"]),
        "proposal_count": int(proposal_diag["proposal_count"]),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(results),
        "stable_count": sum(1 for item in results if item["expected_mode"] == "stable"),
        "conflict_count": sum(1 for item in results if item["expected_mode"] == "conflict"),
    }
    for mode in ("stable", "conflict"):
        subset = [item for item in results if item["expected_mode"] == mode]
        if not subset:
            continue
        summary[f"{mode}_mean_pressure"] = sum(item["mean_contradiction_pressure"] for item in subset) / len(subset)
        summary[f"{mode}_mean_viability"] = sum(item["mean_viability"] for item in subset) / len(subset)
        summary[f"{mode}_mean_dead_end"] = sum(item["dead_end_count"] for item in subset) / len(subset)
        summary[f"{mode}_mean_proposals"] = sum(item["proposal_count"] for item in subset) / len(subset)
    if "stable_mean_pressure" in summary and "conflict_mean_pressure" in summary:
        summary["pressure_gap_conflict_minus_stable"] = (
            summary["conflict_mean_pressure"] - summary["stable_mean_pressure"]
        )
    if "stable_mean_viability" in summary and "conflict_mean_viability" in summary:
        summary["viability_gap_conflict_minus_stable"] = (
            summary["conflict_mean_viability"] - summary["stable_mean_viability"]
        )
    return summary


def build_markdown_report(
    model_name: str,
    device: str,
    max_length: int,
    seed: int,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Qwen Anchor Probe Report",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Max length: `{max_length}`",
        f"Seed: `{seed}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Stable cases: `{summary['stable_count']}`",
        f"- Conflict cases: `{summary['conflict_count']}`",
    ]
    if "pressure_gap_conflict_minus_stable" in summary:
        lines.append(
            f"- Conflict minus stable pressure gap: `{summary['pressure_gap_conflict_minus_stable']:.4f}`"
        )
    if "viability_gap_conflict_minus_stable" in summary:
        lines.append(
            f"- Conflict minus stable viability gap: `{summary['viability_gap_conflict_minus_stable']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Case table",
            "",
            "| Case | Expected | Tokens | Active | Pressure | Viability | Dead ends | Proposals |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        lines.append(
            "| {name} | {expected_mode} | {tokens} | {num_active} | {mean_contradiction_pressure:.4f} | "
            "{mean_viability:.4f} | {dead_end_count} | {proposal_count} |".format(**item)
        )
    lines.extend(["", "## Interpretation", ""])
    pressure_gap = summary.get("pressure_gap_conflict_minus_stable")
    viability_gap = summary.get("viability_gap_conflict_minus_stable")
    lines.append("- This report is diagnostic only; it does not yet apply proposal-guided decoding.")
    if pressure_gap is not None:
        if pressure_gap > 0:
            lines.append(
                f"- Conflict-tagged prompts show higher contradiction pressure than stable prompts in this run (`+{pressure_gap:.4f}`), which is the direction we want."
            )
        else:
            lines.append(
                f"- Conflict-tagged prompts did not separate cleanly on contradiction pressure in this run (`{pressure_gap:.4f}`); threshold calibration is still needed."
            )
    if viability_gap is not None:
        if viability_gap < 0:
            lines.append(
                f"- Conflict-tagged prompts show lower viability than stable prompts in this run (`{viability_gap:.4f}`), which is also the intended direction."
            )
        else:
            lines.append(
                f"- Viability did not yet separate cleanly in this run (`{viability_gap:.4f}`), so the current overlay should be treated as exploratory."
            )
    lines.append("- Proposal counts remaining at zero indicate that proposal-path activation is still unresolved in the current overlay.")
    return "\n".join(lines) + "\n"


def write_outputs(
    output_json: Path,
    output_md: Path,
    payload: dict[str, Any],
    report_text: str,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(report_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anchor diagnostics on top of Qwen hidden states.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_probe_results.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_probe_report.md",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.20,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.25,
        anchor_dead_end_threshold=0.40,
    )
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model,
        cfg=cfg,
        device=args.device,
        torch_dtype=torch.float16 if "cuda" in args.device else None,
    )
    overlay.eval()

    print("=== Qwen Anchor Probe ===")
    print(f"model={args.model}")
    print(f"device={args.device}")
    print()

    results: list[dict[str, Any]] = []
    for case in make_qwen_probe_cases():
        result = collect_case_result(
            overlay=overlay,
            case_name=case.name,
            case_description=case.description,
            case_prompt=case.prompt,
            expected_mode=case.expected_mode,
            max_length=args.max_length,
        )
        results.append(result)
        print(f"--- {case.name} ---")
        print(f"description={case.description}")
        print(f"expected_mode={case.expected_mode}")
        print(f"tokens={result['tokens']}")
        print(f"num_active={result['num_active']}")
        print(f"mean_contradiction_pressure={result['mean_contradiction_pressure']:.4f}")
        print(f"mean_viability={result['mean_viability']:.4f}")
        print(f"dead_end_count={result['dead_end_count']}")
        print(f"proposal_count={result['proposal_count']}")
        print()

    summary = summarize_results(results)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "seed": args.seed,
        "results": results,
        "summary": summary,
    }
    report_text = build_markdown_report(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        seed=args.seed,
        results=results,
        summary=summary,
    )
    write_outputs(
        output_json=args.output_json,
        output_md=args.output_md,
        payload=payload,
        report_text=report_text,
    )
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
