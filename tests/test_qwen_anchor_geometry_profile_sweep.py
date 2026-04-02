from __future__ import annotations

from scripts.run_qwen_anchor_geometry_profile_sweep import (
    build_markdown_report,
    summarize_profile,
)


def test_summarize_profile_extracts_crystallization_layers() -> None:
    aggregate = {
        "tokenization_summary": {"clean_case_count": 12, "noisy_case_count": 1, "skip_case_count": 0},
        "modes": {
            "full_span": {
                "clean_only": {
                    "verdict": "partial_separation",
                    "case_count": 12,
                    "max_separation_layer": {"layer": 24},
                    "first_positive_layer": 20,
                    "stable_birth_layer": 22,
                    "group_aggregates": {
                        "demo": {
                            "anchor_class": "content_like",
                            "transition_summary": {
                                "transitional": True,
                                "sign_changes": 1,
                                "first_content_like_layer": 20,
                                "first_procedure_like_layer": 27,
                                "strongest_content_like_layer": 24,
                                "strongest_procedure_like_layer": 29,
                            },
                        }
                    },
                },
                "all_valid": {
                    "verdict": "partial_separation",
                    "case_count": 13,
                    "max_separation_layer": {"layer": 24},
                    "first_positive_layer": 20,
                    "stable_birth_layer": 22,
                    "group_aggregates": {},
                },
            },
            "trimmed_span": {
                "clean_only": {
                    "verdict": "clear_separation",
                    "case_count": 12,
                    "max_separation_layer": {"layer": 26},
                    "first_positive_layer": 21,
                    "stable_birth_layer": 23,
                    "group_aggregates": {
                        "demo": {
                            "anchor_class": "content_like",
                            "transition_summary": {
                                "transitional": False,
                                "sign_changes": 0,
                                "first_content_like_layer": 21,
                                "first_procedure_like_layer": None,
                                "strongest_content_like_layer": 26,
                                "strongest_procedure_like_layer": None,
                            },
                        }
                    },
                },
                "all_valid": {
                    "verdict": "clear_separation",
                    "case_count": 13,
                    "max_separation_layer": {"layer": 26},
                    "first_positive_layer": 21,
                    "stable_birth_layer": 23,
                    "group_aggregates": {},
                },
            },
        },
    }
    interpretation = {
        "support_after_tokenization_controls": "supported",
    }

    class _Case:
        def __init__(self, name: str, anchor_text: str) -> None:
            self.name = name
            self.anchor_text = anchor_text
            self.anchor_class = "content_like"
            self.anchor_group = "demo"
            self.prompt = "demo prompt"
            self.description = "demo"

    summary = summarize_profile(
        profile_name="medium",
        cases=[_Case("a", "x y z"), _Case("b", "x y z q")],
        results=[],
        aggregate=aggregate,
        interpretation=interpretation,
    )
    assert summary["trimmed_clean"]["stable_birth_layer"] == 23
    assert summary["trimmed_clean"]["max_separation_layer"] == 26
    assert summary["trimmed_group_transitions"]["demo"]["transitional"] is False


def test_build_markdown_report_mentions_crystallization_layers() -> None:
    report = build_markdown_report(
        model_name="Qwen/Qwen3.5-4B",
        device="cuda",
        layers=list(range(32)),
        profile_summaries=[
            {
                "profile": "short",
                "token_count_stats": {"mean": 4.0},
                "tokenization_summary": {"clean_case_count": 12, "noisy_case_count": 1},
                "interpretation": {"support_after_tokenization_controls": "supported"},
                "trimmed_clean": {
                    "verdict": "clear_separation",
                    "first_positive_layer": 21,
                    "stable_birth_layer": 23,
                    "max_separation_layer": 27,
                },
                "trimmed_group_transitions": {
                    "demo": {
                        "anchor_class": "content_like",
                        "transitional": True,
                        "sign_changes": 1,
                        "first_content_like_layer": 21,
                        "first_procedure_like_layer": 28,
                        "strongest_content_like_layer": 27,
                        "strongest_procedure_like_layer": 30,
                    }
                },
            }
        ],
    )
    assert "Crystallization summary by anchor span profile" in report
    assert "stable birth layer" in report
    assert "| short |" in report
