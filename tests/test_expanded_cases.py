"""Tests for expanded geometry cases and auto-calibration."""
from __future__ import annotations

import pytest

from src.data.qwen_anchor_geometry_cases import (
    ANCHOR_SPAN_PROFILES,
    ANCHOR_TEXT_BY_GROUP,
    make_qwen_anchor_geometry_cases,
)
from src.utils.anchor_geometry import auto_calibrate_thresholds


class TestExpandedCases:
    def test_case_count_increased(self):
        cases = make_qwen_anchor_geometry_cases(anchor_span_profile="medium")
        assert len(cases) >= 29, f"Expected >= 29 cases, got {len(cases)}"

    def test_all_profiles_produce_same_count(self):
        counts = {
            p: len(make_qwen_anchor_geometry_cases(anchor_span_profile=p))
            for p in ANCHOR_SPAN_PROFILES
        }
        assert len(set(counts.values())) == 1, f"Profile counts differ: {counts}"

    def test_new_groups_in_anchor_text_map(self):
        new_groups = [
            "penicillin_allergy_treatment_protocol",
            "gdpr_data_retention_compliance_policy",
            "mathematical_induction_proof_steps",
            "sql_foreign_key_constraint_enforcement",
            "thread_safe_singleton_initialization_pattern",
            "idempotent_rest_api_retry_policy",
            "recursive_tree_traversal_procedure",
            "strict_typescript_null_safety_policy",
        ]
        for group in new_groups:
            assert group in ANCHOR_TEXT_BY_GROUP, f"Missing group: {group}"
            for profile in ANCHOR_SPAN_PROFILES:
                assert profile in ANCHOR_TEXT_BY_GROUP[group], \
                    f"Missing profile {profile} for {group}"

    def test_anchor_text_in_prompt(self):
        for profile in ANCHOR_SPAN_PROFILES:
            cases = make_qwen_anchor_geometry_cases(anchor_span_profile=profile)
            for case in cases:
                assert case.anchor_text.lower() in case.prompt.lower(), \
                    f"Anchor text '{case.anchor_text}' not in prompt for {case.name} ({profile})"

    def test_unique_names(self):
        cases = make_qwen_anchor_geometry_cases(anchor_span_profile="medium")
        names = [c.name for c in cases]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_anchor_classes_valid(self):
        cases = make_qwen_anchor_geometry_cases(anchor_span_profile="medium")
        for case in cases:
            assert case.anchor_class in ("content_like", "procedure_like"), \
                f"Invalid anchor_class '{case.anchor_class}' for {case.name}"

    def test_new_groups_have_cases(self):
        cases = make_qwen_anchor_geometry_cases(anchor_span_profile="medium")
        groups_with_cases = {c.anchor_group for c in cases}
        for group in ANCHOR_TEXT_BY_GROUP:
            assert group in groups_with_cases, f"No cases for group: {group}"


class TestAutoCalibration:
    def test_basic_clustering(self):
        # Three well-separated clusters
        r1_refs = [0.8, 0.85, 0.9, 0.3, 0.25, 0.35, 0.2, 0.15, 0.1]
        deltas = [0.01, 0.02, 0.01, 0.15, 0.12, 0.18, 0.01, 0.02, 0.01]
        result = auto_calibrate_thresholds(r1_refs, deltas)

        assert result["method"] == "kmeans_3"
        assert result["n_samples"] == 9
        assert 0.0 < result["mature_r1_threshold"] < 1.0
        assert len(result["cluster_labels"]) == 9
        assert set(result["cluster_labels"]) <= {"mature", "template", "flat"}

    def test_fallback_on_insufficient_data(self):
        result = auto_calibrate_thresholds([0.5, 0.6], [0.1, 0.2])
        assert result["method"] == "fallback_default"
        assert result["mature_r1_threshold"] == 0.65
        assert result["template_delta_threshold"] == 0.08

    def test_deterministic(self):
        r1 = [0.8, 0.3, 0.1, 0.9, 0.25, 0.15]
        dt = [0.01, 0.15, 0.02, 0.01, 0.12, 0.01]
        r1_a = auto_calibrate_thresholds(r1, dt, seed=42)
        r1_b = auto_calibrate_thresholds(r1, dt, seed=42)
        assert r1_a["mature_r1_threshold"] == r1_b["mature_r1_threshold"]
        assert r1_a["cluster_labels"] == r1_b["cluster_labels"]

    def test_cluster_centers_returned(self):
        r1 = [0.9, 0.85, 0.1, 0.15, 0.5, 0.45]
        dt = [0.01, 0.02, 0.01, 0.02, 0.15, 0.12]
        result = auto_calibrate_thresholds(r1, dt)
        assert len(result["cluster_centers"]) == 3
        labels_in_centers = {c["label"] for c in result["cluster_centers"]}
        assert labels_in_centers == {"mature", "template", "flat"}
