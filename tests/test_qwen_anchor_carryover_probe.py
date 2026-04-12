from __future__ import annotations

from scripts.run_qwen_anchor_carryover_probe import (
    build_markdown_report,
    summarize_delta_matrix,
)
from src.data.qwen_anchor_carryover_cases import make_qwen_anchor_carryover_cases


def test_make_qwen_anchor_carryover_cases_respects_profiles() -> None:
    short_case = make_qwen_anchor_carryover_cases(anchor_span_profile="short")[0]
    long_case = make_qwen_anchor_carryover_cases(anchor_span_profile="long")[0]
    assert "vegan meal plan policy" in short_case.anchored_prefix
    assert "strictly vegan meal plan policy for every guest" in long_case.anchored_prefix
    assert short_case.anchored_prefix != long_case.anchored_prefix


def test_summarize_delta_matrix_extracts_peak_and_last_token_delta() -> None:
    summary = summarize_delta_matrix(
        layers=[20, 21],
        anchored_matrix=[[0.1, 0.3], [0.5, 0.7]],
        neutral_matrix=[[0.0, 0.2], [0.1, 0.4]],
    )
    assert summary["peak_delta_layer"] == 21
    assert summary["peak_delta_token_index"] == 0
    assert abs(float(summary["peak_delta_value"]) - 0.4) < 1e-6
    assert abs(float(summary["mean_delta_last_token"]) - 0.2) < 1e-6


def test_build_markdown_report_lists_carryover_row() -> None:
    report = build_markdown_report(
        model_name="Qwen/Qwen3.5-4B",
        device="cuda",
        profile_payloads=[
            {
                "profile": "medium",
                "cases": [
                    {
                        "name": "carryover_demo",
                        "anchor_group": "demo_group",
                        "delta_summary": {
                            "peak_delta_layer": 28,
                            "peak_delta_token_index": 2,
                            "peak_delta_value": 0.18,
                            "mean_delta_last_token": 0.07,
                        },
                        "figure_relpath": "figures/qwen_anchor_carryover_probe/demo.png",
                    }
                ],
            }
        ],
    )
    assert "Qwen Anchor Carryover Probe" in report
    assert "| carryover_demo |" in report
    assert "figures/qwen_anchor_carryover_probe/demo.png" in report
