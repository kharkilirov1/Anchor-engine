from __future__ import annotations

from scripts.run_qwen_anchor_layer_profile_map import (
    build_markdown_report,
    summarize_case_profile,
)


def _make_result() -> dict[str, object]:
    return {
        "name": "demo_case",
        "anchor_class": "content_like",
        "anchor_group": "demo_group",
        "status": "ok",
        "modes": {
            "full_span": {
                "status": "ok",
                "span": {"token_count": 5},
                "layer_results": [
                    {"layer": 20, "metrics": {"rank1_explained_variance": 0.2, "adjacent_cosine_coherence": 0.4, "path_tortuosity": 1.2}},
                    {"layer": 21, "metrics": {"rank1_explained_variance": 0.5, "adjacent_cosine_coherence": 0.6, "path_tortuosity": 0.9}},
                ],
            },
            "trimmed_span": {
                "status": "ok",
                "span": {"token_count": 4},
                "layer_results": [
                    {"layer": 20, "metrics": {"rank1_explained_variance": 0.1, "adjacent_cosine_coherence": 0.2, "path_tortuosity": 1.1}},
                    {"layer": 21, "metrics": {"rank1_explained_variance": 0.7, "adjacent_cosine_coherence": 0.5, "path_tortuosity": 0.8}},
                ],
            },
        },
    }


def test_summarize_case_profile_extracts_peaks() -> None:
    summary = summarize_case_profile(_make_result())
    assert summary["mode_summaries"]["full_span"]["rank1_peak_layer"] == 21
    assert summary["mode_summaries"]["full_span"]["coherence_peak_layer"] == 21
    assert summary["mode_summaries"]["full_span"]["tortuosity_min_layer"] == 21
    assert summary["mode_summaries"]["trimmed_span"]["rank1_peak_layer"] == 21


def test_build_markdown_report_lists_case_rows() -> None:
    case_summary = summarize_case_profile(_make_result())
    case_summary["figure_relpath"] = "figures/qwen_anchor_layer_profiles/demo.png"
    report = build_markdown_report(
        model_name="Qwen/Qwen3.5-4B",
        device="cuda",
        profiles=[
            {
                "profile": "medium",
                "case_summaries": [case_summary],
            }
        ],
    )
    assert "Qwen Anchor Layer Profile Map" in report
    assert "| demo_case |" in report
    assert "figures/qwen_anchor_layer_profiles/demo.png" in report
