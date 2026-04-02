from __future__ import annotations

import pytest

from scripts.run_qwen_anchor_length_sweep import build_length_sweep_markdown
from src.data.qwen_anchor_geometry_cases import (
    list_anchor_span_profiles,
    make_qwen_anchor_geometry_cases,
)


def test_make_qwen_anchor_geometry_cases_supports_length_profiles() -> None:
    short_cases = make_qwen_anchor_geometry_cases(anchor_span_profile="short")
    medium_cases = make_qwen_anchor_geometry_cases(anchor_span_profile="medium")
    long_cases = make_qwen_anchor_geometry_cases(anchor_span_profile="long")

    assert len(short_cases) == len(medium_cases) == len(long_cases) == 13
    assert short_cases[0].name == medium_cases[0].name == long_cases[0].name == "content_vegan_brief"
    assert short_cases[0].anchor_text != medium_cases[0].anchor_text
    assert medium_cases[0].anchor_text != long_cases[0].anchor_text
    assert len(short_cases[0].anchor_text.split()) < len(long_cases[0].anchor_text.split())


def test_make_qwen_anchor_geometry_cases_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError):
        make_qwen_anchor_geometry_cases(anchor_span_profile="giant")


def test_list_anchor_span_profiles_is_stable() -> None:
    assert list_anchor_span_profiles() == ("short", "medium", "long")


def test_build_length_sweep_markdown_includes_curve_table() -> None:
    report = build_length_sweep_markdown(
        model_name="Qwen/Qwen3.5-4B",
        device="cuda",
        profile_summaries=[
            {
                "profile": "short",
                "token_count_stats": {"mean": 4.5, "min": 4, "max": 5},
                "best_policy": "flat_failure_gated",
                "best_policy_stats": {"delta_vs_always_base": 0.1, "wins_over_base": 1, "losses_vs_base": 0},
                "clean_base_observed_separation": False,
                "cluster_counts": {"mature": 1, "template": 3, "flat": 9},
                "best_candidate": {
                    "reference_layers": {"mature_layer": 24},
                    "thresholds": {"mature_r1_threshold": 0.4, "template_delta_threshold": 0.02},
                },
                "flat_cluster_summary": {"mean_constraint_delta": 0.2, "rescue_rate": 0.4},
            },
            {
                "profile": "long",
                "token_count_stats": {"mean": 8.5, "min": 8, "max": 9},
                "best_policy": "always_base",
                "best_policy_stats": {"delta_vs_always_base": 0.0, "wins_over_base": 0, "losses_vs_base": 0},
                "clean_base_observed_separation": True,
                "cluster_counts": {"mature": 3, "template": 3, "flat": 7},
                "best_candidate": {
                    "reference_layers": {"mature_layer": 28},
                    "thresholds": {"mature_r1_threshold": 0.31, "template_delta_threshold": 0.01},
                },
                "flat_cluster_summary": {"mean_constraint_delta": -0.14, "rescue_rate": 0.0},
            },
        ],
    )
    assert "Length-to-quality overview" in report
    assert "| short |" in report
    assert "| long |" in report
    assert "Best searched configuration per profile" in report
