from __future__ import annotations

import torch

from src.utils.anchor_geometry import (
    compute_cross_prompt_stability,
    compute_geometry_metrics,
    match_anchor_span,
    select_representative_layers,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.surface_to_ids = {
            "strictly": [11],
            " vegan": [12],
            " meal": [13],
            " plan": [14],
        }
        self.ids_to_surface = {
            101: "<bos>",
            11: "strictly",
            12: " vegan",
            13: " meal",
            14: " plan",
        }

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        _ = add_special_tokens
        if text not in self.surface_to_ids:
            return {"input_ids": []}
        return {"input_ids": list(self.surface_to_ids[text])}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(self.ids_to_surface.get(int(token_id), "") for token_id in token_ids)


def test_select_representative_layers_matches_qwen_style_targets() -> None:
    assert select_representative_layers(28) == [7, 14, 21, 27]


def test_match_anchor_span_prefers_offsets_when_available() -> None:
    text = "The retreat brief requires a strictly vegan meal plan for every guest."
    anchor_text = "strictly vegan meal plan"
    input_ids = [101, 11, 12, 13, 14]
    offsets = [(0, 0), (29, 37), (37, 43), (43, 48), (48, 53)]
    match = match_anchor_span(
        text=text,
        anchor_text=anchor_text,
        input_ids=input_ids,
        tokenizer=DummyTokenizer(),
        offsets=offsets,
    )
    assert match is not None
    assert match.token_start == 1
    assert match.token_end == 4
    assert match.match_method == "offset_mapping"


def test_compute_geometry_metrics_separates_straight_and_curved_paths() -> None:
    straight = torch.tensor(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    curved = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    straight_metrics = compute_geometry_metrics(straight)
    curved_metrics = compute_geometry_metrics(curved)
    assert straight_metrics["adjacent_cosine_coherence"] is not None
    assert curved_metrics["adjacent_cosine_coherence"] is not None
    assert straight_metrics["adjacent_cosine_coherence"] > curved_metrics["adjacent_cosine_coherence"]
    assert straight_metrics["path_tortuosity"] is not None
    assert curved_metrics["path_tortuosity"] is not None
    assert straight_metrics["path_tortuosity"] < curved_metrics["path_tortuosity"]
    assert straight_metrics["rank1_explained_variance"] is not None
    assert curved_metrics["rank1_explained_variance"] is not None
    assert straight_metrics["rank1_explained_variance"] > curved_metrics["rank1_explained_variance"]


def test_compute_cross_prompt_stability_reports_pairwise_cosines() -> None:
    directions = [
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.9, 0.1], dtype=torch.float32),
        torch.tensor([0.8, 0.2], dtype=torch.float32),
    ]
    stability = compute_cross_prompt_stability(directions)
    assert stability["pair_count"] == 3
    assert stability["mean_pairwise_cosine"] is not None
    assert stability["mean_pairwise_cosine"] > 0.9
