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

    hyp_def_injection = {
        "script": "run_qwen_injection_geometry_probe.py",
        "default_args": {"anchor_profile": "medium"},
    }
    injection_command = orchestrate.build_worker_command(hyp_def_injection, state)
    assert "--profile" in injection_command
    assert "--anchor-profile" not in injection_command


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


def test_build_worker_command_preserves_underscore_flags_for_selected_scripts() -> None:
    state = {"model": "Qwen/Qwen3.5-4B"}

    carryover_command = orchestrate.build_worker_command(
        {
            "script": "run_qwen_anchor_carryover_probe.py",
            "default_args": {"profiles": ["medium"], "case_name": "procedure_contradiction_proof"},
        },
        state,
    )
    assert "--case_name" in carryover_command
    assert "--case-name" not in carryover_command
    assert "--model" in carryover_command

    future_command = orchestrate.build_worker_command(
        {
            "script": "run_qwen_future_influence_probe.py",
            "default_args": {"max_length": 128, "future_window": 8, "top_k": 5},
        },
        state,
    )
    assert "--max_length" in future_command
    assert "--future_window" in future_command
    assert "--top_k" in future_command
    assert "--model" in future_command


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


def test_analyzer_parse_result_falls_back_to_worker_result_json(monkeypatch) -> None:
    monkeypatch.setitem(
        orchestrate.EXPERIMENT_REGISTRY,
        "TEST_REMOTE_DIAG",
        {
            "description": "test",
            "phase": 1,
            "script": "run_qwen_per_case_diagnostic_v2.py",
            "default_args": {"profile": "medium"},
            "output_pattern": "archive/*.json",
            "result_key": "spearman_rho",
            "success_threshold": 0.4,
            "depends_on": [],
        },
    )

    summary = orchestrate.analyzer_parse_result(
        "TEST_REMOTE_DIAG",
        {
            "status": "success",
            "result_json": {
                "spearman_rho": 0.6,
                "n_valid": 5,
                "n_total": 6,
                "profile": "medium",
                "mean_tr": 3.4,
                "mean_cs": 0.4,
            },
        },
    )

    assert summary["output_file"] is None
    assert abs(float(summary["metric_value"]) - 0.6) < 1e-6
    assert summary["profile"] == "medium"


def test_analyzer_parse_result_synthesizes_concept_direction_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "qwen_anchor_concept_direction_map.json"
    output_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile": "medium",
                        "cases": [
                            {"heatmap_summary": {"peak_layer": 7, "peak_cosine": 0.31}},
                            {"heatmap_summary": {"peak_layer": 9, "peak_cosine": 0.41}},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setitem(
        orchestrate.EXPERIMENT_REGISTRY,
        "TEST_CONCEPT_DIRECTION",
        {
            "description": "test",
            "phase": 1,
            "script": "run_qwen_anchor_concept_direction_map.py",
            "default_args": {"profiles": ["medium"]},
            "output_pattern": "archive/*.json",
            "result_key": "summary.mean_peak_cosine",
            "success_threshold": None,
            "depends_on": [],
        },
    )

    summary = orchestrate.analyzer_parse_result(
        "TEST_CONCEPT_DIRECTION",
        {"status": "success", "output_file": str(output_path)},
    )

    assert summary["output_file"] == output_path.name
    assert abs(float(summary["metric_value"]) - 0.36) < 1e-6
