from __future__ import annotations

from scripts.run_abpt_autoresearch_loop import (
    build_default_specs,
    pick_next_spec,
    summarize_history,
)


def test_summarize_history_prefers_best_val_bpb() -> None:
    summary = summarize_history(
        [
            {"loss": 5.0, "bpb": 7.2, "val_bpb": 7.1},
            {"loss": 4.8, "bpb": 7.0, "val_bpb": 6.9},
            {"loss": 4.9, "bpb": 7.1, "val_bpb": 7.0},
        ]
    )

    assert summary["best_val_bpb"] == 6.9
    assert summary["best_loss"] == 4.8


def test_pick_next_spec_starts_with_anchor_synthetic_plain_baseline() -> None:
    spec = pick_next_spec(build_default_specs(), [])
    assert spec is not None
    assert spec.dataset == "anchor-synthetic"
    assert spec.stage == "a"
    assert spec.preset == "baseline-0"
