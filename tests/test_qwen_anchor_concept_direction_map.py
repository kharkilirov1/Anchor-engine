from __future__ import annotations

import torch

from scripts.run_qwen_anchor_concept_direction_map import (
    _concept_pairwise_cosines,
    build_markdown_report,
)


def test_concept_pairwise_cosines_is_symmetric() -> None:
    payload = _concept_pairwise_cosines(
        concept_vectors={
            20: {
                "group_a": torch.tensor([1.0, 0.0]),
                "group_b": torch.tensor([0.0, 1.0]),
            }
        }
    )
    assert payload[20]["group_a"]["group_a"] == 1.0
    assert payload[20]["group_b"]["group_b"] == 1.0
    assert abs(float(payload[20]["group_a"]["group_b"])) < 1e-6
    assert payload[20]["group_a"]["group_b"] == payload[20]["group_b"]["group_a"]


def test_build_markdown_report_lists_case_figures() -> None:
    report = build_markdown_report(
        model_name="Qwen/Qwen3.5-4B",
        device="cuda",
        profile_payloads=[
            {
                "profile": "medium",
                "groups": ["demo_group"],
                "cases": [
                    {
                        "name": "demo_case",
                        "anchor_class": "content_like",
                        "anchor_group": "demo_group",
                        "heatmap_summary": {
                            "peak_layer": 21,
                            "peak_token_index": 3,
                            "peak_cosine": 0.42,
                        },
                        "figure_relpath": "figures/qwen_anchor_concept_direction_map/demo.png",
                    }
                ],
                "concept_vector_norms": [
                    {"layer": 20, "demo_group": 0.25},
                    {"layer": 21, "demo_group": 0.50},
                ],
            }
        ],
    )
    assert "Qwen Anchor Concept Direction Map" in report
    assert "| demo_case |" in report
    assert "figures/qwen_anchor_concept_direction_map/demo.png" in report
