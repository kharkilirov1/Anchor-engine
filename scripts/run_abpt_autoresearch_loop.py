from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_module
from src.model.config import PRESETS, ModelConfig


@dataclass(frozen=True)
class ABPTExperimentSpec:
    name: str
    preset: str
    stage: str
    dataset: str
    steps: int
    batch_size: int
    eval_interval: int
    seq_len: int
    tinystories_bytes: int = 120_000
    tinystories_vocab_size: int = 256
    seed: int = 42

    def slug(self) -> str:
        return (
            f"{self.name}_{self.dataset}_stage_{self.stage}_"
            f"{self.preset.replace('-', '_')}_steps_{self.steps}_seed_{self.seed}"
        )

    def model_key(self) -> str:
        return f"{self.stage}:{self.preset}"

    def dataset_key(self) -> str:
        if self.dataset == "tinystories-bpe":
            return f"{self.dataset}:{self.tinystories_bytes}"
        return self.dataset


def build_default_specs() -> list[ABPTExperimentSpec]:
    specs: list[ABPTExperimentSpec] = [
        ABPTExperimentSpec("plain_baseline", "baseline-0", "a", "anchor-synthetic", 18, 8, 6, 24),
        ABPTExperimentSpec("attnres_baseline", "baseline-1-attnres", "a", "anchor-synthetic", 18, 8, 6, 24),
        ABPTExperimentSpec("branches_only", "baseline-2-branches", "a", "anchor-synthetic", 18, 8, 6, 24),
        ABPTExperimentSpec("plastic_only", "baseline-3-plastic", "a", "anchor-synthetic", 18, 8, 6, 24),
        ABPTExperimentSpec("full_stage_a", "full", "a", "anchor-synthetic", 18, 8, 6, 24),
        ABPTExperimentSpec("full_stage_b", "full", "b", "anchor-synthetic", 18, 8, 6, 24),
        ABPTExperimentSpec("anchor_stage", "full", "anchor", "anchor-synthetic", 18, 8, 6, 24),
        ABPTExperimentSpec("plain_baseline", "baseline-0", "a", "tinystories-bpe", 8, 4, 4, 32),
        ABPTExperimentSpec("attnres_baseline", "baseline-1-attnres", "a", "tinystories-bpe", 8, 4, 4, 32),
        ABPTExperimentSpec("branches_only", "baseline-2-branches", "a", "tinystories-bpe", 8, 4, 4, 32),
        ABPTExperimentSpec("plastic_only", "baseline-3-plastic", "a", "tinystories-bpe", 8, 4, 4, 32),
        ABPTExperimentSpec("full_stage_a", "full", "a", "tinystories-bpe", 8, 4, 4, 32),
        ABPTExperimentSpec("full_stage_b", "full", "b", "tinystories-bpe", 8, 4, 4, 32),
        ABPTExperimentSpec("anchor_stage", "full", "anchor", "tinystories-bpe", 8, 4, 4, 32),
    ]
    return specs


def _safe_float(value: object, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_history(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {
            "final_loss": float("inf"),
            "final_bpb": float("inf"),
            "final_val_bpb": float("inf"),
            "best_val_bpb": float("inf"),
            "best_loss": float("inf"),
        }
    final = history[-1]
    best_val = min((_safe_float(step.get("val_bpb")) for step in history), default=float("inf"))
    best_loss = min((_safe_float(step.get("loss")) for step in history), default=float("inf"))
    return {
        "final_loss": _safe_float(final.get("loss")),
        "final_bpb": _safe_float(final.get("bpb")),
        "final_val_bpb": _safe_float(final.get("val_bpb")),
        "best_val_bpb": best_val,
        "best_loss": best_loss,
    }


def run_spec(spec: ABPTExperimentSpec, output_dir: Path) -> dict[str, Any]:
    torch.manual_seed(spec.seed)
    random.seed(spec.seed)

    cfg: ModelConfig = replace(PRESETS[spec.preset])
    cfg.batch_size = spec.batch_size
    cfg.max_steps = spec.steps
    cfg.eval_interval = spec.eval_interval
    cfg.max_seq_len = spec.seq_len

    history_path = output_dir / f"{spec.slug()}.history.json"
    model = train_module.train(
        cfg=cfg,
        device="cpu",
        stage=spec.stage,
        data_dir=str(ROOT / "data_cache"),
        dataset=spec.dataset,
        history_path=str(history_path),
        tinystories_bytes=spec.tinystories_bytes,
        tinystories_vocab_size=spec.tinystories_vocab_size,
    )
    history = model.training_history
    summary = summarize_history(history)
    params = sum(p.numel() for p in model.parameters())
    result = {
        "spec": asdict(spec),
        "param_count": params,
        "history_path": str(history_path),
        "history_len": len(history),
        "summary": summary,
    }
    result_path = output_dir / f"{spec.slug()}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def load_results(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "loop_state.json":
            continue
        try:
            payload = _read_json(path)
        except json.JSONDecodeError:
            continue
        if "spec" in payload and "summary" in payload:
            payload["__path__"] = str(path)
            results.append(payload)
    return results


def load_loop_history(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "loop_state.json"
    if not path.exists():
        return []
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return []
    iterations = payload.get("iterations", [])
    return iterations if isinstance(iterations, list) else []


def expected_dataset_coverage(specs: list[ABPTExperimentSpec]) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for spec in specs:
        coverage.setdefault(spec.model_key(), set()).add(spec.dataset)
    return coverage


def aggregate_leaderboard(
    results: list[dict[str, Any]],
    specs: list[ABPTExperimentSpec] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    dataset_coverage: dict[str, set[str]] = {}
    expected_coverage = expected_dataset_coverage(specs or [])
    for result in results:
        spec = result["spec"]
        model_key = f"{spec['stage']}:{spec['preset']}"
        grouped.setdefault(model_key, []).append(_safe_float(result["summary"].get("best_val_bpb")))
        dataset_coverage.setdefault(model_key, set()).add(spec["dataset"])

    rows: list[dict[str, Any]] = []
    for model_key, scores in grouped.items():
        seen_datasets = sorted(dataset_coverage.get(model_key, set()))
        target_datasets = sorted(expected_coverage.get(model_key, set()))
        rows.append(
            {
                "model_key": model_key,
                "mean_best_val_bpb": fmean(scores),
                "runs": len(scores),
                "datasets": seen_datasets,
                "coverage": f"{len(seen_datasets)}/{max(len(target_datasets), 1)}",
                "complete": set(seen_datasets) == set(target_datasets) if target_datasets else True,
            }
        )
    rows.sort(key=lambda row: (not row["complete"], row["mean_best_val_bpb"]))
    return rows


def write_leaderboard(
    output_dir: Path,
    results: list[dict[str, Any]],
    specs: list[ABPTExperimentSpec],
) -> Path:
    leaderboard = aggregate_leaderboard(results, specs)
    lines = [
        "| Model | Mean Best Val BPB | Runs | Coverage | Datasets |",
        "|---|---:|---:|---:|---|",
    ]
    for row in leaderboard:
        lines.append(
            f"| {row['model_key']} | {row['mean_best_val_bpb']:.4f} | "
            f"{row['runs']} | {row['coverage']} | {', '.join(row['datasets'])} |"
        )
    path = output_dir / "leaderboard.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def pick_next_spec(specs: list[ABPTExperimentSpec], results: list[dict[str, Any]]) -> ABPTExperimentSpec | None:
    if not results:
        for spec in specs:
            if spec.dataset == "anchor-synthetic" and spec.stage == "a" and spec.preset == "baseline-0":
                return spec
    seen = {json.dumps(result["spec"], sort_keys=True) for result in results}
    leaderboard = aggregate_leaderboard(results, specs)
    best_models = {row["model_key"] for row in leaderboard[:3]}
    dataset_counts = {"anchor-synthetic": 0, "tinystories-bpe": 0}
    expected_coverage = expected_dataset_coverage(specs)
    existing_coverage: dict[str, set[str]] = {}
    for result in results:
        dataset_counts[result["spec"]["dataset"]] = dataset_counts.get(result["spec"]["dataset"], 0) + 1
        model_key = f"{result['spec']['stage']}:{result['spec']['preset']}"
        existing_coverage.setdefault(model_key, set()).add(result["spec"]["dataset"])

    candidates: list[tuple[float, ABPTExperimentSpec]] = []
    for spec in specs:
        if json.dumps(asdict(spec), sort_keys=True) in seen:
            continue
        score = 5.0
        score += 0.6 if dataset_counts.get(spec.dataset, 0) <= min(dataset_counts.values()) else 0.0
        if spec.model_key() in best_models:
            score += 0.4
            if spec.dataset in (expected_coverage.get(spec.model_key(), set()) - existing_coverage.get(spec.model_key(), set())):
                score += 0.35
        if spec.stage == "anchor":
            score += 0.15
        if spec.dataset == "anchor-synthetic":
            score += 0.25
        if spec.steps <= 10:
            score += 0.05
        candidates.append((score, spec))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].dataset == "tinystories-bpe"), reverse=True)
    return candidates[0][1]


def save_loop_state(output_dir: Path, history: list[dict[str, Any]]) -> Path:
    path = output_dir / "loop_state.json"
    path.write_text(
        json.dumps(
            {"iterations": history, "updated_at": datetime.now(timezone.utc).isoformat()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def run_loop(output_dir: Path, iterations: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_default_specs()
    completed = load_results(output_dir)
    loop_history: list[dict[str, Any]] = load_loop_history(output_dir)

    for idx in range(iterations):
        spec = pick_next_spec(specs, completed)
        if spec is None:
            break
        result = run_spec(spec, output_dir)
        completed.append(result)
        loop_history.append(
            {
                "iteration": len(loop_history) + 1,
                "spec": asdict(spec),
                "result_path": result["result_path"],
                "best_val_bpb": result["summary"]["best_val_bpb"],
            }
        )
        save_loop_state(output_dir, loop_history)
        write_leaderboard(output_dir, completed, specs)

    leaderboard = aggregate_leaderboard(completed, specs)
    summary = {
        "output_dir": str(output_dir),
        "iterations_ran": len(loop_history),
        "leaderboard": leaderboard,
    }
    summary_path = output_dir / "final_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact ABPT autoresearch loop on CPU.")
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "abpt_autoresearch"))
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()

    summary = run_loop(Path(args.output_dir), iterations=args.iterations)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
