from __future__ import annotations

from src.utils.autoresearch import (
    DEFAULT_FRONTIER_MODELS,
    build_frontier_specs,
    choose_next_experiment,
    build_global_leaderboard,
)


def _fake_result(
    *,
    dataset: str,
    steps: int,
    time_budget_s: float,
    models: tuple[str, ...],
    repo_id: str = "bigcode/the-stack-smol-xs",
    lang: str = "python",
    target_bytes: int = 1_200_000,
) -> dict[str, object]:
    dataset_block: dict[str, object]
    if dataset == "code":
        dataset_block = {
            "name": "the-stack-bpe",
            "repo_id": repo_id,
            "lang": lang,
            "target_bytes": target_bytes,
        }
    else:
        dataset_block = {
            "name": "roneneldan/TinyStories",
            "train_rows": 1000,
            "val_rows": 100,
        }

    payload: dict[str, object] = {
        "dataset": dataset_block,
        "runtime": {
            "steps": steps,
            "time_budget_s": time_budget_s,
            "models": list(models),
        },
        "uniform": {
            "best_by_val_loss": {"val_loss": 6.0},
            "best_by_val_acc": {"val_accuracy": 0.06},
            "final": {"val_loss": 6.05, "train_tokens_per_s": 1000.0},
        },
    }
    for model in models:
        if model == "uniform":
            continue
        payload[model] = {
            "best_by_val_loss": {"val_loss": 5.9 if model == "structured_v2" else 6.02},
            "best_by_val_acc": {"val_accuracy": 0.07 if model == "structured_v2" else 0.055},
            "final": {"val_loss": 5.94 if model == "structured_v2" else 6.08, "train_tokens_per_s": 800.0},
        }
    return payload


def test_build_frontier_specs_has_cross_domain_coverage() -> None:
    specs = build_frontier_specs()
    names = {spec.name for spec in specs}
    assert names == {
        "tinystories_equal_step_frontier",
        "tinystories_equal_time_frontier",
        "code_equal_step_frontier",
        "code_equal_time_frontier",
    }


def test_choose_next_experiment_prefers_larger_pending_gap() -> None:
    specs = build_frontier_specs(frontier_models=DEFAULT_FRONTIER_MODELS)
    results = [
        _fake_result(
            dataset="tinystories",
            steps=150,
            time_budget_s=0.0,
            models=("uniform", "structured_runtime", "structured_v2", "structured_fast"),
        ),
        _fake_result(
            dataset="tinystories",
            steps=9999,
            time_budget_s=20.0,
            models=("uniform", "structured_runtime", "structured_v2"),
        ),
        _fake_result(
            dataset="code",
            steps=100,
            time_budget_s=0.0,
            models=("uniform", "structured_runtime", "structured_copy"),
        ),
    ]
    decision = choose_next_experiment(specs=specs, results=results)
    assert decision is not None
    assert decision.spec.name in {"code_equal_time_frontier", "code_equal_step_frontier"}
    assert "structured_v2" in decision.pending_models


def test_build_global_leaderboard_ranks_better_model_first() -> None:
    rows = build_global_leaderboard(
        [
            _fake_result(
                dataset="tinystories",
                steps=150,
                time_budget_s=0.0,
                models=("uniform", "structured_v2", "structured_runtime"),
            ),
            _fake_result(
                dataset="code",
                steps=100,
                time_budget_s=0.0,
                models=("uniform", "structured_v2"),
            ),
        ],
        frontier_models=("uniform", "structured_v2", "structured_runtime"),
    )
    assert rows[0]["model"] == "structured_v2"
