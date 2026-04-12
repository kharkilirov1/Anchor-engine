from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
import json


DEFAULT_FRONTIER_MODELS: tuple[str, ...] = (
    "uniform",
    "structured_runtime",
    "structured_v2",
    "structured_v2_copy",
    "structured_fast",
    "structured_copy",
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: str
    models: tuple[str, ...]
    steps: int
    batch_size: int
    seq_len: int
    eval_every: int
    eval_steps: int
    train_rows: int = 1000
    val_rows: int = 100
    time_budget_s: float = 0.0
    code_repo: str = "bigcode/the-stack-smol-xs"
    code_lang: str = "python"
    code_bytes: int = 1_200_000
    seed: int = 42

    def dataset_key(self) -> str:
        if self.dataset == "code":
            return f"code::{self.code_repo}::{self.code_lang}::{self.code_bytes}"
        return f"tinystories::{self.train_rows}::{self.val_rows}"

    def budget_key(self) -> str:
        if self.time_budget_s > 0.0:
            rounded = int(round(self.time_budget_s))
            return f"time::{rounded}s"
        return f"steps::{self.steps}"

    def slug(self) -> str:
        if self.time_budget_s > 0.0:
            rounded = int(round(self.time_budget_s))
            suffix = f"time_{rounded}s"
        else:
            suffix = f"steps_{self.steps}"
        return f"{self.name}_{suffix}_seed_{self.seed}"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionDecision:
    spec: ExperimentSpec
    pending_models: tuple[str, ...]
    score: float
    reason: str


def build_frontier_specs(
    frontier_models: tuple[str, ...] = DEFAULT_FRONTIER_MODELS,
    tinystories_steps: int = 150,
    code_steps: int = 100,
    time_budget_s: float = 20.0,
    batch_size: int = 4,
    seq_len: int = 64,
    eval_every: int = 25,
    eval_steps: int = 5,
    train_rows: int = 1000,
    val_rows: int = 100,
    code_repo: str = "bigcode/the-stack-smol-xs",
    code_lang: str = "python",
    code_bytes: int = 1_200_000,
    seeds: tuple[int, ...] = (42,),
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for seed in seeds:
        specs.extend(
            [
                ExperimentSpec(
                    name="tinystories_equal_step_frontier",
                    dataset="tinystories",
                    models=frontier_models,
                    steps=tinystories_steps,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    eval_every=eval_every,
                    eval_steps=eval_steps,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    seed=seed,
                ),
                ExperimentSpec(
                    name="tinystories_equal_time_frontier",
                    dataset="tinystories",
                    models=frontier_models,
                    steps=9999,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    eval_every=eval_every,
                    eval_steps=eval_steps,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    time_budget_s=time_budget_s,
                    seed=seed,
                ),
                ExperimentSpec(
                    name="code_equal_step_frontier",
                    dataset="code",
                    models=frontier_models,
                    steps=code_steps,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    eval_every=eval_every,
                    eval_steps=eval_steps,
                    code_repo=code_repo,
                    code_lang=code_lang,
                    code_bytes=code_bytes,
                    seed=seed,
                ),
                ExperimentSpec(
                    name="code_equal_time_frontier",
                    dataset="code",
                    models=frontier_models,
                    steps=9999,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    eval_every=eval_every,
                    eval_steps=eval_steps,
                    time_budget_s=time_budget_s,
                    code_repo=code_repo,
                    code_lang=code_lang,
                    code_bytes=code_bytes,
                    seed=seed,
                ),
            ]
        )
    return specs


def is_benchmark_result(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if "dataset" not in payload or "runtime" not in payload:
        return False
    dataset = payload.get("dataset")
    runtime = payload.get("runtime")
    return isinstance(dataset, dict) and isinstance(runtime, dict)


def load_benchmark_results(results_dir: Path) -> list[dict[str, object]]:
    loaded: list[dict[str, object]] = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not is_benchmark_result(payload):
            continue
        item = dict(payload)
        item["__path__"] = str(path)
        loaded.append(item)
    return loaded


def result_dataset_key(result: dict[str, object]) -> str:
    dataset = result["dataset"]
    assert isinstance(dataset, dict)
    dataset_name = str(dataset.get("name", ""))
    if dataset_name == "the-stack-bpe":
        return (
            f"code::{dataset.get('repo_id')}::{dataset.get('lang')}::"
            f"{dataset.get('target_bytes')}"
        )
    return f"tinystories::{dataset.get('train_rows')}::{dataset.get('val_rows')}"


def result_budget_key(result: dict[str, object]) -> str:
    runtime = result["runtime"]
    assert isinstance(runtime, dict)
    time_budget_s = float(runtime.get("time_budget_s", 0.0))
    if time_budget_s > 0.0:
        rounded = int(round(time_budget_s))
        return f"time::{rounded}s"
    return f"steps::{int(runtime.get('steps', 0))}"


def result_models(result: dict[str, object]) -> tuple[str, ...]:
    runtime = result["runtime"]
    assert isinstance(runtime, dict)
    models = runtime.get("models", [])
    return tuple(str(model) for model in models)


def result_covers_spec(result: dict[str, object], spec: ExperimentSpec) -> bool:
    runtime = result["runtime"]
    assert isinstance(runtime, dict)
    return (
        result_dataset_key(result) == spec.dataset_key()
        and result_budget_key(result) == spec.budget_key()
        and int(runtime.get("seed", 42)) == spec.seed
    )


def find_pending_models(spec: ExperimentSpec, results: list[dict[str, object]]) -> tuple[str, ...]:
    pending: list[str] = []
    for model in spec.models:
        if model == "uniform":
            continue
        covered = any(result_covers_spec(result, spec) and model in result_models(result) for result in results)
        if not covered:
            pending.append(model)
    return tuple(pending)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def model_relative_score(result: dict[str, object], model_name: str) -> float | None:
    baseline = result.get("uniform")
    model = result.get(model_name)
    if not isinstance(baseline, dict) or not isinstance(model, dict):
        return None

    baseline_best = baseline.get("best_by_val_loss")
    model_best = model.get("best_by_val_loss")
    baseline_acc = baseline.get("best_by_val_acc")
    model_acc = model.get("best_by_val_acc")
    baseline_final = baseline.get("final")
    model_final = model.get("final")
    if not all(isinstance(item, dict) for item in (baseline_best, model_best, baseline_acc, model_acc, baseline_final, model_final)):
        return None

    loss_gain = _safe_float(baseline_best.get("val_loss")) - _safe_float(model_best.get("val_loss"))
    acc_gain = _safe_float(model_acc.get("val_accuracy")) - _safe_float(baseline_acc.get("val_accuracy"))
    baseline_speed = _safe_float(baseline_final.get("train_tokens_per_s"), 1.0)
    model_speed = _safe_float(model_final.get("train_tokens_per_s"), 0.0)
    speed_ratio = model_speed / max(baseline_speed, 1e-8)
    baseline_stability = _safe_float(baseline_final.get("val_loss")) - _safe_float(baseline_best.get("val_loss"))
    model_stability = _safe_float(model_final.get("val_loss")) - _safe_float(model_best.get("val_loss"))
    stability_gain = baseline_stability - model_stability

    return 4.0 * loss_gain + 2.0 * acc_gain + 0.5 * (speed_ratio - 1.0) + stability_gain


def aggregate_model_promise(
    results: list[dict[str, object]],
    frontier_models: tuple[str, ...] = DEFAULT_FRONTIER_MODELS,
) -> dict[str, float]:
    aggregated: dict[str, list[float]] = {model: [] for model in frontier_models if model != "uniform"}
    for result in results:
        for model in aggregated:
            score = model_relative_score(result, model)
            if score is not None:
                aggregated[model].append(score)
    return {
        model: fmean(scores) if scores else 0.0
        for model, scores in aggregated.items()
    }


def coverage_counts(results: list[dict[str, object]]) -> dict[str, int]:
    counts = {"tinystories": 0, "code": 0, "time": 0, "steps": 0}
    for result in results:
        dataset_key = result_dataset_key(result)
        budget_key = result_budget_key(result)
        if dataset_key.startswith("code::"):
            counts["code"] += 1
        else:
            counts["tinystories"] += 1
        if budget_key.startswith("time::"):
            counts["time"] += 1
        else:
            counts["steps"] += 1
    return counts


def choose_next_experiment(
    specs: list[ExperimentSpec],
    results: list[dict[str, object]],
) -> SelectionDecision | None:
    decisions = rank_candidate_experiments(specs=specs, results=results)
    return decisions[0] if decisions else None


def rank_candidate_experiments(
    specs: list[ExperimentSpec],
    results: list[dict[str, object]],
) -> list[SelectionDecision]:
    promise = aggregate_model_promise(results)
    counts = coverage_counts(results)
    recent_names: list[str] = []
    for result in results[-12:]:
        runtime = result.get("runtime")
        dataset = result.get("dataset")
        if not isinstance(runtime, dict) or not isinstance(dataset, dict):
            continue
        if str(dataset.get("name")) == "the-stack-bpe":
            prefix = "code"
        else:
            prefix = "tinystories"
        mode = "time" if float(runtime.get("time_budget_s", 0.0)) > 0.0 else "steps"
        recent_names.append(f"{prefix}_{mode}")
    decisions: list[SelectionDecision] = []

    for spec in specs:
        pending = find_pending_models(spec, results)
        if not pending:
            continue
        recent_penalty = 0.25 if f"{spec.dataset}_{'time' if spec.time_budget_s > 0.0 else 'steps'}" in recent_names[-4:] else 0.0
        dataset_bonus = 0.35 if (
            spec.dataset == "code" and counts["code"] <= counts["tinystories"]
        ) or (
            spec.dataset == "tinystories" and counts["tinystories"] < counts["code"]
        ) else 0.0
        budget_bonus = 0.2 if (
            spec.time_budget_s > 0.0 and counts["time"] <= counts["steps"]
        ) or (
            spec.time_budget_s == 0.0 and counts["steps"] < counts["time"]
        ) else 0.0
        promise_bonus = sum(max(0.0, promise.get(model, 0.0)) for model in pending)
        score = 3.0 * len(pending) + promise_bonus + dataset_bonus + budget_bonus - recent_penalty
        reason = (
            f"pending={', '.join(pending)}; "
            f"dataset_bonus={dataset_bonus:.2f}; "
            f"budget_bonus={budget_bonus:.2f}; "
            f"promise_bonus={promise_bonus:.3f}; "
            f"recent_penalty={recent_penalty:.2f}"
        )
        decisions.append(
            SelectionDecision(
                spec=spec,
                pending_models=pending,
                score=score,
                reason=reason,
            )
        )

    decisions.sort(
        key=lambda decision: (
            decision.score,
            len(decision.pending_models),
            1 if decision.spec.dataset == "code" else 0,
            1 if decision.spec.time_budget_s > 0.0 else 0,
            -decision.spec.seed,
        ),
        reverse=True,
    )
    return decisions


def build_command(
    spec: ExperimentSpec,
    runner_path: Path,
    output_path: Path,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        str(runner_path),
        "--dataset",
        spec.dataset,
        "--steps",
        str(spec.steps),
        "--batch_size",
        str(spec.batch_size),
        "--seq_len",
        str(spec.seq_len),
        "--eval_every",
        str(spec.eval_every),
        "--eval_steps",
        str(spec.eval_steps),
        "--seed",
        str(spec.seed),
        "--output",
        str(output_path),
        "--models",
        *spec.models,
    ]
    if spec.time_budget_s > 0.0:
        command.extend(["--time_budget_s", str(spec.time_budget_s)])
    if spec.dataset == "code":
        command.extend(
            [
                "--code_repo",
                spec.code_repo,
                "--code_lang",
                spec.code_lang,
                "--code_bytes",
                str(spec.code_bytes),
            ]
        )
    else:
        command.extend(
            [
                "--train_rows",
                str(spec.train_rows),
                "--val_rows",
                str(spec.val_rows),
            ]
        )
    return command


def build_global_leaderboard(
    results: list[dict[str, object]],
    frontier_models: tuple[str, ...] = DEFAULT_FRONTIER_MODELS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    promise = aggregate_model_promise(results, frontier_models=frontier_models)
    for model in frontier_models:
        if model == "uniform":
            continue
        coverage = 0
        exact_hits: list[str] = []
        for result in results:
            if model in result_models(result):
                coverage += 1
                exact_hits.append(f"{result_dataset_key(result)}|{result_budget_key(result)}")
        rows.append(
            {
                "model": model,
                "mean_relative_score": promise.get(model, 0.0),
                "coverage_runs": coverage,
                "coverage_keys": sorted(set(exact_hits)),
            }
        )
    rows.sort(key=lambda row: (float(row["mean_relative_score"]), int(row["coverage_runs"])), reverse=True)
    return rows


def leaderboard_markdown(rows: list[dict[str, object]]) -> str:
    header = [
        "| Model | Mean Relative Score | Coverage Runs | Coverage Keys |",
        "|---|---:|---:|---|",
    ]
    body: list[str] = []
    for row in rows:
        coverage_keys = "<br>".join(str(item) for item in row["coverage_keys"])
        body.append(
            f"| {row['model']} | {float(row['mean_relative_score']):+.4f} | "
            f"{int(row['coverage_runs'])} | {coverage_keys} |"
        )
    return "\n".join(header + body)
