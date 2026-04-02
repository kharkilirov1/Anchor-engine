from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_probe_cases import QwenProbeCase, make_qwen_probe_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from scripts.run_qwen_anchor_probe import summarize_results


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def pairwise_family_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, dict[str, Any]]] = {}
    for item in results:
        families.setdefault(item["family"], {})[item["expected_mode"]] = item

    metrics: dict[str, dict[str, Any]] = {}
    for family, pair in families.items():
        stable = pair.get("stable")
        conflict = pair.get("conflict")
        if stable is None or conflict is None:
            continue
        pressure_gap = conflict["mean_contradiction_pressure"] - stable["mean_contradiction_pressure"]
        viability_gap = stable["mean_viability"] - conflict["mean_viability"]
        metrics[family] = {
            "pressure_gap": pressure_gap,
            "viability_gap": viability_gap,
            "pressure_win": pressure_gap > 0.0,
            "viability_win": viability_gap > 0.0,
            "joint_win": pressure_gap > 0.0 and viability_gap > 0.0,
        }
    return metrics


def score_configuration(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_results(results)
    family_metrics = pairwise_family_metrics(results)
    pressure_wins = sum(int(item["pressure_win"]) for item in family_metrics.values())
    viability_wins = sum(int(item["viability_win"]) for item in family_metrics.values())
    joint_wins = sum(int(item["joint_win"]) for item in family_metrics.values())
    pressure_gap = float(summary.get("pressure_gap_conflict_minus_stable", 0.0))
    viability_gap = float(summary.get("stable_mean_viability", 0.0) - summary.get("conflict_mean_viability", 0.0))
    score = (
        2.0 * pressure_wins
        + 2.0 * viability_wins
        + 3.0 * joint_wins
        + 10.0 * max(pressure_gap, 0.0)
        + 10.0 * max(viability_gap, 0.0)
    )
    return {
        "score": score,
        "summary": summary,
        "family_metrics": family_metrics,
        "pressure_wins": pressure_wins,
        "viability_wins": viability_wins,
        "joint_wins": joint_wins,
        "pressure_gap": pressure_gap,
        "viability_gap": viability_gap,
    }


def cache_hidden_cases(
    overlay: QwenAnchorOverlay,
    cases: list[QwenProbeCase],
    max_length: int,
) -> list[dict[str, Any]]:
    cached: list[dict[str, Any]] = []
    for case in cases:
        encoded = overlay.tokenizer(
            [case.prompt],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(overlay.parameters()).device
        batch = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = overlay.extract_hidden_batch(**batch)
        cached.append(
            {
                "case": case,
                "input_ids": batch["input_ids"],
                "attention_mask": batch.get("attention_mask"),
                "hidden": hidden,
            }
        )
    return cached


def evaluate_configuration(
    base_model: torch.nn.Module,
    tokenizer: Any,
    hidden_cache: list[dict[str, Any]],
    anchor_threshold: float,
    revision_threshold: float,
    contradiction_threshold: float,
    dead_end_threshold: float,
    init_seed: int,
) -> dict[str, Any]:
    torch.manual_seed(init_seed)
    cfg = replace(
        TOY_CONFIG,
        d_model=int(getattr(base_model.config, "hidden_size")),
        vocab_size=int(getattr(base_model.config, "vocab_size")),
        max_seq_len=int(getattr(base_model.config, "max_position_embeddings", getattr(base_model.config, "max_seq_len", 32768))),
        anchor_threshold=anchor_threshold,
        anchor_revision_threshold=revision_threshold,
        anchor_contradiction_threshold=contradiction_threshold,
        anchor_dead_end_threshold=dead_end_threshold,
    )
    overlay = QwenAnchorOverlay(base_model=base_model, cfg=cfg, tokenizer=tokenizer)
    overlay = overlay.to(next(base_model.parameters()).device)
    overlay.eval()

    results: list[dict[str, Any]] = []
    for item in hidden_cache:
        case: QwenProbeCase = item["case"]
        out = overlay.analyze_hidden_batch(
            hidden=item["hidden"],
            input_ids=item["input_ids"],
            attention_mask=item["attention_mask"],
        )
        diag = out["anchor_diagnostics"]
        proposal_diag = out["proposal_diagnostics"]
        results.append(
            {
                "name": case.name,
                "family": case.family,
                "description": case.description,
                "expected_mode": case.expected_mode,
                "tokens": int(item["input_ids"].numel()),
                "num_active": int(diag["num_active"]),
                "mean_contradiction_pressure": float(diag["mean_contradiction_pressure"]),
                "mean_viability": float(diag["mean_viability"]),
                "dead_end_count": int(diag["dead_end_count"]),
                "proposal_count": int(proposal_diag["proposal_count"]),
            }
        )

    scored = score_configuration(results)
    return {
        "config": {
            "anchor_threshold": anchor_threshold,
            "anchor_revision_threshold": revision_threshold,
            "anchor_contradiction_threshold": contradiction_threshold,
            "anchor_dead_end_threshold": dead_end_threshold,
        },
        "results": results,
        **scored,
    }


def build_markdown_report(
    model_name: str,
    device: str,
    seed: int,
    max_length: int,
    evaluations: list[dict[str, Any]],
) -> str:
    best = evaluations[0]
    lines = [
        "# Qwen Anchor Threshold Calibration",
        "",
        f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        f"Seed: `{seed}`",
        f"Max length: `{max_length}`",
        f"Sweeps evaluated: `{len(evaluations)}`",
        "",
        "## Best configuration",
        "",
        f"- anchor_threshold: `{best['config']['anchor_threshold']:.2f}`",
        f"- anchor_revision_threshold: `{best['config']['anchor_revision_threshold']:.2f}`",
        f"- anchor_contradiction_threshold: `{best['config']['anchor_contradiction_threshold']:.2f}`",
        f"- anchor_dead_end_threshold: `{best['config']['anchor_dead_end_threshold']:.2f}`",
        f"- score: `{best['score']:.4f}`",
        f"- pressure wins: `{best['pressure_wins']}`",
        f"- viability wins: `{best['viability_wins']}`",
        f"- joint wins: `{best['joint_wins']}`",
        f"- pressure gap: `{best['pressure_gap']:.4f}`",
        f"- viability gap: `{best['viability_gap']:.4f}`",
        "",
        "## Top configurations",
        "",
        "| Rank | anchor | revise | contradiction | dead_end | score | pressure wins | viability wins | joint wins | pressure gap | viability gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(evaluations[:10], start=1):
        cfg = item["config"]
        lines.append(
            f"| {rank} | {cfg['anchor_threshold']:.2f} | {cfg['anchor_revision_threshold']:.2f} | "
            f"{cfg['anchor_contradiction_threshold']:.2f} | {cfg['anchor_dead_end_threshold']:.2f} | "
            f"{item['score']:.4f} | {item['pressure_wins']} | {item['viability_wins']} | {item['joint_wins']} | "
            f"{item['pressure_gap']:.4f} | {item['viability_gap']:.4f} |"
        )

    lines.extend(["", "## Best family breakdown", "", "| Family | Pressure gap | Viability gap | Pressure win | Viability win | Joint win |", "|---|---:|---:|---|---|---|"])
    for family, metrics in best["family_metrics"].items():
        lines.append(
            f"| {family} | {metrics['pressure_gap']:.4f} | {metrics['viability_gap']:.4f} | "
            f"{metrics['pressure_win']} | {metrics['viability_win']} | {metrics['joint_win']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Calibration currently optimizes stable-vs-conflict separation on the fixed prompt suite.",
            "- This is still a probe-time heuristic, not a learned objective.",
            "- Proposal counts remain zero in the current overlay, so threshold tuning only improves detector/viability behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep anchor thresholds on cached Qwen hidden states.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--anchor_thresholds", type=str, default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--revision_thresholds", type=str, default="0.35,0.45,0.55")
    parser.add_argument("--contradiction_thresholds", type=str, default="0.20,0.25,0.30,0.35")
    parser.add_argument("--dead_end_thresholds", type=str, default="0.35,0.40,0.45,0.50")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_anchor_threshold_calibration.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_threshold_calibration.md",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model,
        cfg=replace(
            TOY_CONFIG,
            anchor_threshold=0.20,
            anchor_revision_threshold=0.45,
            anchor_contradiction_threshold=0.25,
            anchor_dead_end_threshold=0.40,
        ),
        device=args.device,
        torch_dtype=torch.float16 if "cuda" in args.device else None,
        low_cpu_mem_usage=True,
    )
    overlay.eval()

    cases = make_qwen_probe_cases()
    hidden_cache = cache_hidden_cases(overlay=overlay, cases=cases, max_length=args.max_length)

    evaluations: list[dict[str, Any]] = []
    for anchor_threshold, revision_threshold, contradiction_threshold, dead_end_threshold in itertools.product(
        parse_float_list(args.anchor_thresholds),
        parse_float_list(args.revision_thresholds),
        parse_float_list(args.contradiction_thresholds),
        parse_float_list(args.dead_end_thresholds),
    ):
        evaluation = evaluate_configuration(
            base_model=overlay.base_model,
            tokenizer=overlay.tokenizer,
            hidden_cache=hidden_cache,
            anchor_threshold=anchor_threshold,
            revision_threshold=revision_threshold,
            contradiction_threshold=contradiction_threshold,
            dead_end_threshold=dead_end_threshold,
            init_seed=args.seed,
        )
        evaluations.append(evaluation)

    evaluations.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "max_length": args.max_length,
        "evaluations": evaluations,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=args.device,
        seed=args.seed,
        max_length=args.max_length,
        evaluations=evaluations,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(report, encoding="utf-8")

    best = evaluations[0]
    print("=== Qwen Anchor Threshold Calibration ===")
    print(f"model={args.model}")
    print(f"device={args.device}")
    print(f"evaluations={len(evaluations)}")
    print(f"best_score={best['score']:.4f}")
    print(f"best_anchor_threshold={best['config']['anchor_threshold']:.2f}")
    print(f"best_revision_threshold={best['config']['anchor_revision_threshold']:.2f}")
    print(f"best_contradiction_threshold={best['config']['anchor_contradiction_threshold']:.2f}")
    print(f"best_dead_end_threshold={best['config']['anchor_dead_end_threshold']:.2f}")
    print(f"best_pressure_wins={best['pressure_wins']}")
    print(f"best_viability_wins={best['viability_wins']}")
    print(f"best_joint_wins={best['joint_wins']}")
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
