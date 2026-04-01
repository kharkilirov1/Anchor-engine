from __future__ import annotations

import pytest

from scripts.run_qwen_geometry_generation_calibration import (
    KEYWORD_MAP,
    build_calibration_summary,
    build_policy_simulation,
    compute_constraint_analysis,
)


def test_compute_constraint_analysis_flags_degenerate_repetition() -> None:
    text = " ".join(["The retreat brief requires a strictly vegan meal plan policy."] * 8)
    analysis = compute_constraint_analysis(
        text=text,
        keyword_spec=KEYWORD_MAP["strictly_vegan_meal_plan_policy"],
    )
    assert analysis["positive_total"] >= 8
    assert analysis["unique_positive_hits"] == 1
    assert analysis["degenerate_output"] is True
    assert analysis["constraint_satisfied"] is False
    assert analysis["constraint_score"] == 0.0


def test_compute_constraint_analysis_keeps_fastapi_thread_text_valid() -> None:
    text = (
        "The async FastAPI router keeps request handling explicit, and asyncio moves heavy work "
        "off the request thread while each endpoint still uses dependency injection."
    )
    analysis = compute_constraint_analysis(
        text=text,
        keyword_spec=KEYWORD_MAP["async_fastapi_service_architecture_policy"],
    )
    assert analysis["negative_total"] == 0
    assert analysis["unique_positive_hits"] >= 4
    assert analysis["degenerate_output"] is False
    assert analysis["constraint_satisfied"] is True


def test_build_calibration_summary_excludes_degenerate_bases() -> None:
    cases = [
        {
            "name": "degenerate_flat",
            "anchor_cluster": "flat",
            "constraint_delta": -1.0,
            "base_degenerate": True,
            "included_in_calibration": False,
            "r1_at_24": 0.45,
            "anchor_analysis": {"drift_detected": False},
        },
        {
            "name": "included_flat",
            "anchor_cluster": "flat",
            "constraint_delta": 1.0,
            "base_degenerate": False,
            "included_in_calibration": True,
            "r1_at_24": 0.52,
            "anchor_analysis": {"drift_detected": False},
        },
        {
            "name": "template_case",
            "anchor_cluster": "template",
            "constraint_delta": 0.0,
            "base_degenerate": False,
            "included_in_calibration": True,
            "r1_at_24": 0.50,
            "anchor_analysis": {"drift_detected": False},
        },
        {
            "name": "mature_case",
            "anchor_cluster": "mature",
            "constraint_delta": 0.0,
            "base_degenerate": False,
            "included_in_calibration": True,
            "r1_at_24": 0.70,
            "anchor_analysis": {"drift_detected": False},
        },
    ]
    summary = build_calibration_summary(cases)
    flat = summary["by_cluster_clean_base"]["flat"]
    flat_rescue = summary["by_cluster_degenerate_base"]["flat"]
    assert summary["excluded_base_degenerate_case_names"] == ["degenerate_flat"]
    assert flat["n_total"] == 2
    assert flat["n_selected"] == 1
    assert flat["mean_constraint_delta"] == 1.0
    assert flat["median_constraint_delta"] == 1.0
    assert flat["excluded_case_names"] == ["degenerate_flat"]
    assert flat_rescue["n_selected"] == 1
    assert flat_rescue["mean_constraint_delta"] == -1.0
    assert summary["threshold_candidates"]["clean_base_observed_separation"] is False


def test_build_policy_simulation_surfaces_flat_failure_gated_rescue() -> None:
    cases = [
        {
            "name": "flat_failed",
            "anchor_cluster": "flat",
            "base_degenerate": True,
            "base_analysis": {"constraint_score": 0.0},
            "anchor_analysis": {"constraint_score": 1.0},
        },
        {
            "name": "template_clean",
            "anchor_cluster": "template",
            "base_degenerate": False,
            "base_analysis": {"constraint_score": 1.0},
            "anchor_analysis": {"constraint_score": 0.0},
        },
        {
            "name": "mature_clean",
            "anchor_cluster": "mature",
            "base_degenerate": False,
            "base_analysis": {"constraint_score": 1.0},
            "anchor_analysis": {"constraint_score": 1.0},
        },
    ]
    simulation = build_policy_simulation(cases)
    all_cases = simulation["all_cases"]
    assert all_cases["always_base"]["mean_constraint_score"] == 2.0 / 3.0
    assert all_cases["always_anchor"]["mean_constraint_score"] == 2.0 / 3.0
    assert all_cases["flat_failure_gated"]["mean_constraint_score"] == 1.0
    assert all_cases["flat_failure_gated"]["delta_vs_always_base"] == pytest.approx(1.0 / 3.0)
    assert all_cases["flat_failure_gated"]["anchor_pick_count"] == 1
    assert simulation["degenerate_base"]["flat_failure_gated"]["mean_constraint_score"] == 1.0
