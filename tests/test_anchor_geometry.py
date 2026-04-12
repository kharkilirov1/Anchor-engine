from __future__ import annotations

import torch

from src.utils.anchor_geometry import (
    build_computability_flags,
    compute_cross_prompt_stability,
    compute_geometry_metrics,
    decode_token_pieces,
    decode_token_surfaces,
    detect_anchor_span,
    list_model_layers,
    match_anchor_span,
    select_representative_layers,
    token_has_leading_whitespace,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.surface_to_ids = {
            "strictly vegan meal plan": [11, 12, 13, 14],
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
        self.converted = {
            101: "<bos>",
            11: "strictly",
            12: "Ġvegan",
            13: "Ġmeal",
            14: "Ġplan",
        }

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        _ = add_special_tokens
        if text not in self.surface_to_ids:
            return {"input_ids": []}
        return {"input_ids": list(self.surface_to_ids[text])}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(self.ids_to_surface.get(int(token_id), "") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.converted[int(token_id)] for token_id in token_ids]


class DummyBatchEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids

    def __getitem__(self, key: str) -> list[int]:
        if key != "input_ids":
            raise KeyError(key)
        return self.input_ids


class DummyTokenizerBatchEncoding(DummyTokenizer):
    def __call__(self, text: str, add_special_tokens: bool = False) -> DummyBatchEncoding:
        _ = add_special_tokens
        return DummyBatchEncoding(list(self.surface_to_ids.get(text, [])))


def test_select_representative_layers_matches_qwen_style_targets() -> None:
    assert select_representative_layers(28) == [7, 14, 21, 27]


def test_list_model_layers_returns_zero_based_stack() -> None:
    assert list_model_layers(4) == [0, 1, 2, 3]


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


def test_match_anchor_span_supports_batch_encoding_like_tokenizer_outputs() -> None:
    match = match_anchor_span(
        text="strictly vegan meal plan",
        anchor_text="strictly vegan meal plan",
        input_ids=[11, 12, 13, 14],
        tokenizer=DummyTokenizerBatchEncoding(),
        offsets=None,
    )
    assert match is not None
    assert match.token_start == 0
    assert match.token_end == 3


def test_decode_token_helpers_and_leading_whitespace_marker() -> None:
    tokenizer = DummyTokenizer()
    token_ids = [11, 12, 13]
    assert decode_token_surfaces(tokenizer, token_ids) == ["strictly", "Ġvegan", "Ġmeal"]
    assert decode_token_pieces(tokenizer, token_ids) == ["strictly", " vegan", " meal"]
    assert token_has_leading_whitespace("Ġvegan", " vegan") is True
    assert token_has_leading_whitespace("strictly", "strictly") is False


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


def test_build_computability_flags_marks_short_spans() -> None:
    metrics = compute_geometry_metrics(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    flags = build_computability_flags(metrics)
    assert flags["span_mean_direction"] is True
    assert flags["mean_direction_norm"] is True
    assert flags["adjacent_cosine_coherence"] is False
    assert flags["path_tortuosity"] is False
    assert flags["rank1_explained_variance"] is False


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


def test_detect_anchor_span_finds_high_attention_region() -> None:
    """Attention concentrated on tokens 3-4 should be detected as anchor."""
    seq_len = 10
    n_heads = 2
    n_layers = 4

    attentions: list[torch.Tensor] = []
    for layer in range(n_layers + 1):
        attn = torch.zeros(1, n_heads, seq_len, seq_len)
        # Last token attends uniformly
        attn[:, :, -1, :] = 1.0 / seq_len
        # Spike attention on tokens 3-4 in later layers
        if layer >= 3:
            attn[:, :, -1, :] = 0.01
            attn[:, :, -1, 3] = 0.45
            attn[:, :, -1, 4] = 0.45
        attentions.append(attn)

    probe_layers = [2, 3]  # attentions index = layer + 1
    result = detect_anchor_span(tuple(attentions), probe_layers)
    assert result is not None
    assert result.match_method == "attention_mass"
    assert result.token_start <= 3
    assert result.token_end >= 4


def test_detect_anchor_span_returns_none_for_short_sequence() -> None:
    attn = torch.zeros(1, 2, 3, 3)
    result = detect_anchor_span((attn,), [0], min_width=2, skip_special=1)
    assert result is None


def test_detect_anchor_span_returns_none_for_empty_inputs() -> None:
    assert detect_anchor_span((), []) is None
    assert detect_anchor_span((), [0, 1]) is None
