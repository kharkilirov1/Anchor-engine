from __future__ import annotations

import pytest

from scripts.run_qwen_geometry_generation_calibration import (
    KEYWORD_MAP,
    apply_cluster_configuration,
    build_calibration_summary,
    build_policy_simulation,
    compute_constraint_analysis,
    search_reference_and_thresholds,
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
            "r1_reference": 0.45,
            "anchor_analysis": {"drift_detected": False},
        },
        {
            "name": "included_flat",
            "anchor_cluster": "flat",
            "constraint_delta": 1.0,
            "base_degenerate": False,
            "included_in_calibration": True,
            "r1_reference": 0.52,
            "anchor_analysis": {"drift_detected": False},
        },
        {
            "name": "template_case",
            "anchor_cluster": "template",
            "constraint_delta": 0.0,
            "base_degenerate": False,
            "included_in_calibration": True,
            "r1_reference": 0.50,
            "anchor_analysis": {"drift_detected": False},
        },
        {
            "name": "mature_case",
            "anchor_cluster": "mature",
            "constraint_delta": 0.0,
            "base_degenerate": False,
            "included_in_calibration": True,
            "r1_reference": 0.70,
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
    assert flat["r1_reference_range"] == [0.45, 0.52]
    assert summary["threshold_candidates"]["clean_base_observed_separation"] is False


def test_build_calibration_summary_uses_passed_reference_layers_and_thresholds() -> None:
    cases = [
        {
            "name": "flat_case",
            "anchor_cluster": "flat",
            "constraint_delta": 0.0,
            "base_degenerate": False,
            "included_in_calibration": True,
            "r1_reference": 0.44,
            "anchor_analysis": {"drift_detected": False},
        }
    ]
    summary = build_calibration_summary(
        cases,
        reference_layers={"mature_layer": 28, "template_prev_layer": 30, "template_curr_layer": 31},
        thresholds={"mature_r1_threshold": 0.51, "template_delta_threshold": 0.07},
    )
    assert summary["reference_layers"]["mature_layer"] == 28
    assert summary["threshold_candidates"]["r1_reference_mature_threshold"] == pytest.approx(0.51)
    assert summary["threshold_candidates"]["delta_template_pair_threshold"] == pytest.approx(0.07)


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


def test_apply_cluster_configuration_assigns_crystallization_clusters() -> None:
    cases = [
        {"name": "mature_case", "rank1_profile": {"10": 0.71, "12": 0.70, "13": 0.72}},
        {"name": "template_case", "rank1_profile": {"10": 0.40, "12": 0.20, "13": 0.55}},
        {"name": "flat_case", "rank1_profile": {"10": 0.41, "12": 0.36, "13": 0.38}},
    ]
    enriched = apply_cluster_configuration(
        cases,
        reference_layers={
            "slope_start_layer": 10,
            "slope_end_layer": 10,
            "mature_layer": 10,
            "template_prev_layer": 12,
            "template_curr_layer": 13,
        },
        mature_threshold=0.65,
        template_threshold=0.15,
    )
    cluster_map = {case["name"]: case["anchor_cluster"] for case in enriched}
    assert cluster_map == {
        "mature_case": "mature",
        "template_case": "template",
        "flat_case": "flat",
    }


def test_search_reference_and_thresholds_finds_model_specific_candidate() -> None:
    cases = [
        {
            "name": "flat_rescue",
            "rank1_profile": {"10": 0.42, "11": 0.41, "12": 0.40, "13": 0.39},
            "base_analysis": {"constraint_score": 0.0},
            "anchor_analysis": {"constraint_score": 1.0, "drift_detected": False},
            "base_degenerate": True,
            "included_in_calibration": False,
            "constraint_delta": 1.0,
        },
        {
            "name": "template_case",
            "rank1_profile": {"10": 0.38, "11": 0.41, "12": 0.20, "13": 0.58},
            "base_analysis": {"constraint_score": 1.0},
            "anchor_analysis": {"constraint_score": 1.0, "drift_detected": False},
            "base_degenerate": False,
            "included_in_calibration": True,
            "constraint_delta": 0.0,
        },
        {
            "name": "mature_case",
            "rank1_profile": {"10": 0.72, "11": 0.73, "12": 0.74, "13": 0.75},
            "base_analysis": {"constraint_score": 1.0},
            "anchor_analysis": {"constraint_score": 1.0, "drift_detected": False},
            "base_degenerate": False,
            "included_in_calibration": True,
            "constraint_delta": 0.0,
        },
    ]
    result = search_reference_and_thresholds(cases, search_layers=[10, 11, 12, 13])
    best = result["candidate"]
    assert result["search_summary"]["n_reference_candidates"] > 0
    assert result["search_summary"]["n_total_candidates"] > 0
    assert set(best["cluster_counts"]) == {"mature", "template", "flat"}
    assert sum(best["cluster_counts"].values()) == 3
    assert len(result["cases"]) == 3
