from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_module
from scripts.run_abpt_autoresearch_loop import summarize_history
from src.model.config import PRESETS, ModelConfig


@dataclass(frozen=True)
class SweepRun:
    name: str
    preset: str
    stage: str
    dataset: str
    steps: int
    batch_size: int
    eval_interval: int
    seq_len: int
    tinystories_bytes: int = 180_000
    tinystories_vocab_size: int = 256
    seed: int = 42

    def slug(self) -> str:
        return (
            f"{self.name}_{self.dataset}_stage_{self.stage}_"
            f"{self.preset.replace('-', '_')}_steps_{self.steps}_seed_{self.seed}"
        )

    def model_key(self) -> str:
        return f"{self.stage}:{self.preset}"


def build_runs(anchor_steps: int, tinystories_steps: int) -> list[SweepRun]:
    models = [
        ("plain_baseline", "baseline-0", "a"),
        ("plastic_only", "baseline-3-plastic", "a"),
        ("anchor_stage", "full", "anchor"),
        ("stage_b_full", "full", "b"),
    ]
    runs: list[SweepRun] = []
    for name, preset, stage in models:
        runs.append(SweepRun(name, preset, stage, "anchor-synthetic", anchor_steps, 8, 6, 24))
        runs.append(SweepRun(name, preset, stage, "tinystories-bpe", tinystories_steps, 4, 4, 32))
    return runs


def run_once(spec: SweepRun, output_dir: Path) -> dict[str, Any]:
    torch.manual_seed(spec.seed)
    cfg: ModelConfig = replace(PRESETS[spec.preset])
    cfg.max_steps = spec.steps
    cfg.batch_size = spec.batch_size
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
    summary = summarize_history(model.training_history)
    result = {
        "spec": asdict(spec),
        "param_count": sum(p.numel() for p in model.parameters()),
        "summary": summary,
        "history_path": str(history_path),
    }
    result_path = output_dir / f"{spec.slug()}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    overall: dict[str, list[float]] = {}
    by_dataset: dict[str, list[tuple[str, float, int]]] = {}
    for result in results:
        spec = result["spec"]
        key = f"{spec['stage']}:{spec['preset']}"
        overall.setdefault(key, []).append(result["summary"]["best_val_bpb"])
        by_dataset.setdefault(spec["dataset"], []).append(
            (key, result["summary"]["best_val_bpb"], result["param_count"])
        )

    lines = ["# ABPT equal-budget sweep", ""]
    lines.append("## Overall mean best val BPB")
    lines.append("")
    lines.append("| Model | Mean Best Val BPB | Runs |")
    lines.append("|---|---:|---:|")
    for key, scores in sorted(overall.items(), key=lambda item: fmean(item[1])):
        lines.append(f"| {key} | {fmean(scores):.4f} | {len(scores)} |")

    for dataset, rows in sorted(by_dataset.items()):
        lines.append("")
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Model | Best Val BPB | Params |")
        lines.append("|---|---:|---:|")
        for key, bpb, params in sorted(rows, key=lambda item: item[1]):
            lines.append(f"| {key} | {bpb:.4f} | {params:,} |")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "results": results,
        "overall": {
            key: {"mean_best_val_bpb": fmean(scores), "runs": len(scores)}
            for key, scores in overall.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run equal-budget ABPT comparison on CPU.")
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "abpt_equal_budget_sweep_v1"))
    parser.add_argument("--anchor_steps", type=int, default=24)
    parser.add_argument("--tinystories_steps", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [run_once(spec, output_dir) for spec in build_runs(args.anchor_steps, args.tinystories_steps)]
    write_summary(output_dir, results)
    print(json.dumps({"output_dir": str(output_dir), "runs": len(results)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
