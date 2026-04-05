from __future__ import annotations

import json
from pathlib import Path

from scripts import orchestrate


def test_nested_get_supports_list_indices() -> None:
    payload = {"summary": {"items": [{"value": 1.25}, {"value": 2.5}]}}
    assert orchestrate._nested_get(payload, "summary.items.1.value") == 2.5
    assert orchestrate._nested_get(payload, "summary.items.2.value") is None


def test_build_worker_command_normalizes_known_script_args() -> None:
    hyp_def = {
        "script": "run_qwen_anchor_geometry_probe.py",
        "default_args": {"anchor_profile": "medium", "max_length": 64},
    }
    state = {"model": "Qwen/Qwen3.5-4B"}
    command = orchestrate.build_worker_command(hyp_def, state)
    assert "--model-name" in command
    assert "--anchor-profile" not in command
    assert "--max-length" in command

    hyp_def_carryover = {
        "script": "run_qwen_anchor_carryover_probe.py",
        "default_args": {"anchor_profile": "medium"},
    }
    carryover_command = orchestrate.build_worker_command(hyp_def_carryover, state)
    assert "--profiles" in carryover_command
    assert "--anchor-profile" not in carryover_command

    hyp_def_layer_profile = {
        "script": "run_qwen_anchor_layer_profile_map.py",
        "default_args": {"anchor_profile": "medium"},
    }
    layer_profile_command = orchestrate.build_worker_command(hyp_def_layer_profile, state)
    assert "--profiles" in layer_profile_command
    assert "--anchor-profile" not in layer_profile_command


def test_build_worker_command_applies_cpu_remote_safe_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_CPU_SPACE", "1")
    hyp_def = {
        "script": "run_qwen_per_case_diagnostic_v2.py",
        "default_args": {"profile": "short"},
    }
    state = {"model": "Qwen/Qwen3.5-4B"}

    command = orchestrate.build_worker_command(hyp_def, state)

    assert "--device" in command
    assert "cpu" in command
    assert "--max-new-tokens" in command
    assert "8" in command
    assert "--group-case-cap" in command
    assert "1" in command


def test_strategist_select_next_skips_missing_script(monkeypatch) -> None:
    original_registry = dict(orchestrate.EXPERIMENT_REGISTRY)
    monkeypatch.setattr(
        orchestrate,
        "EXPERIMENT_REGISTRY",
        {
            "BROKEN": {
                "description": "broken",
                "phase": 1,
                "script": "run_qwen_missing_probe.py",
                "default_args": {},
                "output_pattern": "archive/missing.json",
                "result_key": "summary.value",
                "success_threshold": 0.0,
                "depends_on": [],
            },
            "GOOD": {
                "description": "good",
                "phase": 1,
                "script": "run_qwen_phase_probe.py",
                "default_args": {"anchor_profile": "short", "tau": "0.5"},
                "output_pattern": "archive/*.json",
                "result_key": "summary.value",
                "success_threshold": 0.0,
                "depends_on": [],
            },
        },
    )

    state = {
        "budget_remaining": 1,
        "current_phase": 1,
        "phases": {"phase_1": {"experiments": []}},
    }

    try:
        assert orchestrate.strategist_select_next(state) == "GOOD"
    finally:
        monkeypatch.setattr(orchestrate, "EXPERIMENT_REGISTRY", original_registry)


def test_analyzer_parse_result_uses_worker_output_file_and_synthesizes_carryover_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "qwen_anchor_carryover_probe.json"
    output_path.write_text(
        json.dumps(
            {
                "metadata": {"model_name": "Qwen/Qwen3.5-4B"},
                "profiles": [
                    {
                        "profile": "medium",
                        "cases": [
                            {"delta_summary": {"peak_delta_value": 0.4, "mean_delta_last_token": 0.1}},
                            {"delta_summary": {"peak_delta_value": 0.2, "mean_delta_last_token": 0.3}},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setitem(
        orchestrate.EXPERIMENT_REGISTRY,
        "TEST_CARRYOVER",
        {
            "description": "test",
            "phase": 2,
            "script": "run_qwen_anchor_carryover_probe.py",
            "default_args": {"anchor_profile": "medium"},
            "output_pattern": "archive/*.json",
            "result_key": "summary.mean_last_token_delta",
            "success_threshold": 0.0,
            "depends_on": [],
        },
    )

    summary = orchestrate.analyzer_parse_result(
        "TEST_CARRYOVER",
        {"status": "success", "output_file": str(output_path)},
    )

    assert summary["output_file"] == output_path.name
    assert abs(float(summary["metric_value"]) - 0.2) < 1e-6


def test_analyzer_parse_result_synthesizes_layer_profile_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "qwen_anchor_layer_profile_map.json"
    output_path.write_text(
        json.dumps(
            {
                "metadata": {"model_name": "Qwen/Qwen3.5-4B"},
                "profiles": [
                    {
                        "profile": "medium",
                        "case_summaries": [
                            {
                                "status": "ok",
                                "mode_summaries": {
                                    "trimmed_span": {"rank1_peak_layer": 7},
                                    "full_span": {"rank1_peak_layer": 8},
                                },
                            },
                            {
                                "status": "ok",
                                "mode_summaries": {
                                    "trimmed_span": {"rank1_peak_layer": 9},
                                    "full_span": {"rank1_peak_layer": 10},
                                },
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setitem(
        orchestrate.EXPERIMENT_REGISTRY,
        "TEST_LAYER_PROFILE",
        {
            "description": "test",
            "phase": 1,
            "script": "run_qwen_anchor_layer_profile_map.py",
            "default_args": {"anchor_profile": "medium"},
            "output_pattern": "archive/*.json",
            "result_key": "summary.trimmed_rank1_peak_layer_mean",
            "success_threshold": None,
            "depends_on": [],
        },
    )

    summary = orchestrate.analyzer_parse_result(
        "TEST_LAYER_PROFILE",
        {"status": "success", "output_file": str(output_path)},
    )

    assert summary["output_file"] == output_path.name
    assert abs(float(summary["metric_value"]) - 8.0) < 1e-6
