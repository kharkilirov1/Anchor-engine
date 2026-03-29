from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from src.data.qwen_probe_cases import make_qwen_probe_cases
from src.model.config import TOY_CONFIG
from src.model.future_influence import FutureInfluenceScorer
from src.model.future_span_hints import (
    build_auxiliary_future_proposals,
    build_future_hint_candidates,
    is_informative_hint_text,
    summarize_auxiliary_proposals,
)
from src.model.anchor_types import AnchorRecord, AnchorState
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from scripts.calibrate_qwen_anchor_thresholds import pairwise_family_metrics, score_configuration
from scripts.analyze_qwen_span_misses import build_analysis, classify_family
from scripts.compare_qwen_signal_proxies import compare_payloads
from scripts.evaluate_qwen_auxiliary_proposals import build_auxiliary_report_data
from scripts.evaluate_qwen_auxiliary_revision import build_auxiliary_revision_report_data
from scripts.extract_qwen_future_proposal_hints import extract_hint_candidates
from scripts.run_qwen_future_influence_probe import (
    compute_span_anchor_overlap,
    extract_high_influence_spans,
    summarize_results as summarize_future_results,
)
from scripts.run_qwen_anchor_probe import build_markdown_report, summarize_results


class _DummyConfig:
    def __init__(self, hidden_size: int = 32, vocab_size: int = 97, max_position_embeddings: int = 64):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings


class _DummyOutput:
    def __init__(self, logits: torch.Tensor, hidden_states: tuple[torch.Tensor, ...]):
        self.logits = logits
        self.hidden_states = hidden_states


class _DummyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _DummyConfig()
        self.emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.proj = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> _DummyOutput:
        del attention_mask, return_dict
        hidden = self.emb(input_ids)
        logits = self.proj(hidden)
        hidden_states = (hidden * 0.5, hidden) if output_hidden_states else (hidden,)
        return _DummyOutput(logits=logits, hidden_states=hidden_states)


class _DummyTokenizer:
    def __call__(
        self,
        texts: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        rows = []
        for text in texts:
            ids = [min(ord(ch), 96) % 97 for ch in text[:max_length]]
            rows.append(torch.tensor(ids, dtype=torch.long))
        padded = nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
        mask = (padded != 0).long()
        return {"input_ids": padded, "attention_mask": mask}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr((token_id % 26) + 97) for token_id in token_ids)


def test_qwen_anchor_overlay_forward_returns_diagnostics():
    model = _DummyCausalLM()
    cfg = replace(TOY_CONFIG, anchor_threshold=0.10)
    overlay = QwenAnchorOverlay(base_model=model, cfg=cfg)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 12))
    out = overlay(input_ids)

    assert "logits" in out
    assert "anchor_diagnostics" in out
    assert "proposal_diagnostics" in out
    assert out["logits"].shape[:2] == input_ids.shape


def test_qwen_anchor_overlay_analyze_texts_uses_tokenizer():
    model = _DummyCausalLM()
    tokenizer = _DummyTokenizer()
    overlay = QwenAnchorOverlay(base_model=model, cfg=TOY_CONFIG, tokenizer=tokenizer)

    out, batch = overlay.analyze_texts(["alpha beta", "gamma"], max_length=16)

    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == 2
    assert "anchor_diagnostics" in out


def test_qwen_probe_summary_and_report_include_gaps():
    results = [
        {
            "name": "stable_case",
            "family": "toy",
            "description": "stable",
            "expected_mode": "stable",
            "tokens": 10,
            "num_active": 1,
            "mean_contradiction_pressure": 0.2,
            "mean_viability": 0.6,
            "dead_end_count": 1,
            "proposal_count": 0,
        },
        {
            "name": "conflict_case",
            "family": "toy",
            "description": "conflict",
            "expected_mode": "conflict",
            "tokens": 12,
            "num_active": 2,
            "mean_contradiction_pressure": 0.5,
            "mean_viability": 0.3,
            "dead_end_count": 2,
            "proposal_count": 0,
        },
    ]

    summary = summarize_results(results)
    report = build_markdown_report(
        model_name="Qwen/Qwen2.5-1.5B",
        device="cpu",
        max_length=64,
        seed=7,
        cfg={
            "anchor_threshold": 0.2,
            "anchor_revision_threshold": 0.45,
            "anchor_contradiction_threshold": 0.25,
            "anchor_dead_end_threshold": 0.4,
        },
        results=results,
        summary=summary,
    )

    assert summary["pressure_gap_conflict_minus_stable"] > 0
    assert "Qwen Anchor Probe Report" in report
    assert "conflict_case" in report


def test_qwen_probe_cases_are_balanced_by_family():
    cases = make_qwen_probe_cases()

    assert len(cases) >= 12

    by_family: dict[str, set[str]] = {}
    for case in cases:
        by_family.setdefault(case.family, set()).add(case.expected_mode)

    assert by_family
    assert all(modes == {"stable", "conflict"} for modes in by_family.values())


def test_qwen_overlay_can_analyze_hidden_batch_directly():
    model = _DummyCausalLM()
    tokenizer = _DummyTokenizer()
    overlay = QwenAnchorOverlay(base_model=model, cfg=TOY_CONFIG, tokenizer=tokenizer)

    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    hidden = overlay.extract_hidden_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    out = overlay.analyze_hidden_batch(
        hidden=hidden,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert "anchor_diagnostics" in out
    assert "proposal_diagnostics" in out


def test_qwen_calibration_scoring_prefers_family_separation():
    results = [
        {
            "name": "alpha_stable",
            "family": "alpha",
            "description": "stable",
            "expected_mode": "stable",
            "tokens": 10,
            "num_active": 2,
            "mean_contradiction_pressure": 0.20,
            "mean_viability": 0.60,
            "dead_end_count": 1,
            "proposal_count": 0,
        },
        {
            "name": "alpha_conflict",
            "family": "alpha",
            "description": "conflict",
            "expected_mode": "conflict",
            "tokens": 10,
            "num_active": 2,
            "mean_contradiction_pressure": 0.55,
            "mean_viability": 0.30,
            "dead_end_count": 2,
            "proposal_count": 0,
        },
    ]

    family_metrics = pairwise_family_metrics(results)
    scored = score_configuration(results)

    assert family_metrics["alpha"]["pressure_win"] is True
    assert family_metrics["alpha"]["viability_win"] is True
    assert scored["score"] > 0


def test_future_influence_scorer_returns_token_scores():
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    proj = nn.Linear(8, 11)
    logits = proj(hidden)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)

    scorer = FutureInfluenceScorer()
    out = scorer(
        hidden=hidden,
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        future_window=3,
    )

    assert out["scores"].shape == input_ids.shape
    assert out["target_window"] == 3
    assert out["loss"] >= 0.0


def test_qwen_overlay_can_compute_future_influence_from_texts():
    model = _DummyCausalLM()
    tokenizer = _DummyTokenizer()
    overlay = QwenAnchorOverlay(base_model=model, cfg=TOY_CONFIG, tokenizer=tokenizer)

    out, batch = overlay.analyze_texts_with_future_influence(["alpha beta"], max_length=16, future_window=4)

    assert "future_influence" in out
    assert out["future_influence"]["scores"].shape == batch["input_ids"].shape


def test_future_influence_summary_tracks_mode_gap():
    results = [
        {
            "name": "stable_case",
            "family": "toy",
            "description": "stable",
            "expected_mode": "stable",
            "tokens": 8,
            "num_active": 1,
            "mean_contradiction_pressure": 0.2,
            "mean_viability": 0.7,
            "future_loss": 1.0,
            "future_window": 4,
            "mean_future_influence": 0.3,
            "max_future_influence": 1.0,
            "anchor_position_mean_future_influence": 0.35,
            "anchor_positions": [1],
            "top_future_tokens": [],
        },
        {
            "name": "conflict_case",
            "family": "toy",
            "description": "conflict",
            "expected_mode": "conflict",
            "tokens": 8,
            "num_active": 1,
            "mean_contradiction_pressure": 0.4,
            "mean_viability": 0.2,
            "future_loss": 1.3,
            "future_window": 4,
            "mean_future_influence": 0.5,
            "max_future_influence": 1.0,
            "anchor_position_mean_future_influence": 0.55,
            "anchor_positions": [1],
            "top_future_tokens": [],
        },
    ]

    summary = summarize_future_results(results)

    assert summary["future_influence_gap_conflict_minus_stable"] > 0
    assert summary["anchor_future_influence_gap_conflict_minus_stable"] > 0


def test_compare_qwen_signal_proxies_tracks_family_wins():
    anchor_payload = {
        "results": [
            {
                "family": "toy",
                "expected_mode": "stable",
                "mean_contradiction_pressure": 0.2,
                "mean_viability": 0.7,
            },
            {
                "family": "toy",
                "expected_mode": "conflict",
                "mean_contradiction_pressure": 0.4,
                "mean_viability": 0.3,
            },
        ]
    }
    future_payload = {
        "results": [
            {
                "family": "toy",
                "expected_mode": "stable",
                "mean_future_influence": 0.2,
                "anchor_position_mean_future_influence": 0.1,
            },
            {
                "family": "toy",
                "expected_mode": "conflict",
                "mean_future_influence": 0.1,
                "anchor_position_mean_future_influence": 0.5,
            },
        ]
    }

    comparison = compare_payloads(anchor_payload, future_payload)

    assert comparison["summary"]["family_count"] == 1
    assert comparison["summary"]["pressure_wins"] == 1
    assert comparison["summary"]["viability_wins"] == 1
    assert comparison["summary"]["future_wins"] == 0
    assert comparison["summary"]["anchor_future_wins"] == 1


def test_extract_high_influence_spans_groups_contiguous_positions():
    tokenizer = _DummyTokenizer()
    scores = torch.tensor([0.1, 0.8, 0.9, 0.2, 0.85, 0.9], dtype=torch.float32)
    input_ids = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)

    spans = extract_high_influence_spans(
        scores=scores,
        input_ids=input_ids,
        tokenizer=tokenizer,
        min_score=0.75,
        top_spans=4,
    )

    assert len(spans) == 2
    assert spans[0]["start"] == 4
    assert spans[0]["end"] == 5
    assert spans[1]["start"] == 1
    assert spans[1]["end"] == 2


def test_compute_span_anchor_overlap_reports_bidirectional_overlap():
    future_spans = [{"start": 2, "end": 4}, {"start": 8, "end": 9}]
    active_anchor_spans = [{"start": 3, "end": 5}, {"start": 10, "end": 12}]

    overlap = compute_span_anchor_overlap(future_spans, active_anchor_spans)

    assert overlap["future_span_overlap_ratio"] == 0.5
    assert overlap["anchor_span_overlap_ratio"] == 0.5


def test_build_future_hint_candidates_keeps_only_non_overlapping_spans():
    future_spans = [
        {"start": 3, "end": 4, "text": "overlap", "mean_score": 0.8, "max_score": 0.9, "length": 2},
        {"start": 8, "end": 8, "text": "hint", "mean_score": 0.7, "max_score": 0.7, "length": 1},
    ]
    active_anchor_spans = [{"start": 4, "end": 5}]

    hints = build_future_hint_candidates(future_spans, active_anchor_spans)

    assert len(hints) == 1
    assert hints[0]["text"] == "hint"


def test_is_informative_hint_text_filters_stopword_like_spans():
    assert is_informative_hint_text("a witness") is True
    assert is_informative_hint_text("the text starts") is True
    assert is_informative_hint_text(" the ") is False
    assert is_informative_hint_text(".") is False


def test_build_auxiliary_future_proposals_averages_hidden_span():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    hints = [{"start": 0, "end": 1, "mean_score": 0.8, "text": "ab", "max_score": 0.9, "length": 2}]

    proposals = build_auxiliary_future_proposals(
        hidden=hidden,
        input_ids=input_ids,
        future_hint_candidates=hints,
        tokenizer=_DummyTokenizer(),
    )

    assert len(proposals) == 1
    assert proposals[0]["proposal_span"] == (0, 1)
    assert torch.allclose(proposals[0]["repr"], torch.tensor([2.0, 3.0]))


def test_summarize_auxiliary_proposals_reports_counts_and_scores():
    batches = [
        [{"proposal_score": 0.8}, {"proposal_score": 0.4}],
        [],
    ]

    summary = summarize_auxiliary_proposals(batches)

    assert summary["proposal_count"] == 2
    assert summary["batch_with_proposals_count"] == 1
    assert summary["mean_proposal_count_per_batch"] == 1.0
    assert summary["mean_proposal_score"] == pytest.approx(0.6)


def test_classify_family_distinguishes_rescue_patterns():
    assert classify_family(True, True, True) == "aligned"
    assert classify_family(False, True, False) == "future_rescue"
    assert classify_family(True, False, False) == "delta_only"
    assert classify_family(False, False, False) == "both_weak"


def test_build_analysis_summarizes_family_classes():
    anchor_payload = {
        "results": [
            {"family": "alpha", "expected_mode": "stable", "mean_contradiction_pressure": 0.2, "mean_viability": 0.7},
            {"family": "alpha", "expected_mode": "conflict", "mean_contradiction_pressure": 0.5, "mean_viability": 0.2},
            {"family": "beta", "expected_mode": "stable", "mean_contradiction_pressure": 0.4, "mean_viability": 0.3},
            {"family": "beta", "expected_mode": "conflict", "mean_contradiction_pressure": 0.3, "mean_viability": 0.4},
        ]
    }
    future_payload = {
        "results": [
            {"family": "alpha", "expected_mode": "stable", "anchor_position_mean_future_influence": 0.1, "future_span_overlap_ratio": 0.0},
            {"family": "alpha", "expected_mode": "conflict", "anchor_position_mean_future_influence": 0.6, "future_span_overlap_ratio": 0.5},
            {"family": "beta", "expected_mode": "stable", "anchor_position_mean_future_influence": 0.2, "future_span_overlap_ratio": 0.0},
            {"family": "beta", "expected_mode": "conflict", "anchor_position_mean_future_influence": 0.4, "future_span_overlap_ratio": 0.1},
        ]
    }

    analysis = build_analysis(anchor_payload, future_payload)

    assert analysis["summary"]["family_count"] == 2
    assert analysis["summary"]["aligned_count"] == 1
    assert analysis["summary"]["future_rescue_count"] == 1


def test_extract_hint_candidates_prefers_non_overlapping_future_rescue_spans():
    anchor_payload = {
        "results": [
            {"family": "alpha", "expected_mode": "stable", "mean_contradiction_pressure": 0.4, "mean_viability": 0.4},
            {"family": "alpha", "expected_mode": "conflict", "mean_contradiction_pressure": 0.3, "mean_viability": 0.5},
        ]
    }
    future_payload = {
        "results": [
            {
                "family": "alpha",
                "name": "alpha_conflict",
                "expected_mode": "conflict",
                "active_anchor_spans": [{"start": 10, "end": 12}],
                "future_spans": [
                    {"start": 10, "end": 11, "mean_score": 0.8, "max_score": 0.9, "text": "overlap"},
                    {"start": 20, "end": 22, "mean_score": 0.7, "max_score": 0.8, "text": "hint"},
                ],
                "anchor_position_mean_future_influence": 0.7,
                "future_span_overlap_ratio": 0.2,
            }
        ]
    }
    analysis = build_analysis(anchor_payload, {
        "results": [
            {
                "family": "alpha",
                "expected_mode": "stable",
                "anchor_position_mean_future_influence": 0.1,
                "future_span_overlap_ratio": 0.0,
            },
            {
                "family": "alpha",
                "expected_mode": "conflict",
                "anchor_position_mean_future_influence": 0.7,
                "future_span_overlap_ratio": 0.2,
            },
        ]
    })

    hints = extract_hint_candidates(future_payload, analysis)

    assert len(hints) == 1
    assert hints[0]["span_text"] == "hint"
    assert hints[0]["classification"] == "future_rescue"


def test_extract_hint_candidates_prefers_filtered_future_hint_candidates_when_present():
    analysis = {
        "families": [
            {
                "family": "alpha",
                "classification": "future_rescue",
                "anchor_future_gap": 0.5,
            }
        ]
    }
    future_payload = {
        "results": [
            {
                "family": "alpha",
                "name": "alpha_conflict",
                "expected_mode": "conflict",
                "active_anchor_spans": [{"start": 10, "end": 12}],
                "future_hint_candidates": [
                    {"start": 20, "end": 21, "mean_score": 0.9, "max_score": 1.0, "text": "hint"}
                ],
                "future_spans": [
                    {"start": 30, "end": 30, "mean_score": 0.95, "max_score": 0.95, "text": "fallback"}
                ],
            }
        ]
    }

    hints = extract_hint_candidates(future_payload, analysis)

    assert len(hints) == 1
    assert hints[0]["span_text"] == "hint"


def test_build_auxiliary_report_data_tracks_future_rescue_gaps():
    anchor_payload = {
        "results": [
            {"family": "alpha", "expected_mode": "stable", "mean_contradiction_pressure": 0.4, "mean_viability": 0.4},
            {"family": "alpha", "expected_mode": "conflict", "mean_contradiction_pressure": 0.3, "mean_viability": 0.5},
            {"family": "beta", "expected_mode": "stable", "mean_contradiction_pressure": 0.2, "mean_viability": 0.7},
            {"family": "beta", "expected_mode": "conflict", "mean_contradiction_pressure": 0.4, "mean_viability": 0.3},
        ]
    }
    future_payload = {
        "results": [
            {
                "family": "alpha",
                "expected_mode": "stable",
                "anchor_position_mean_future_influence": 0.1,
                "future_span_overlap_ratio": 0.0,
                "auxiliary_proposal_count": 0,
                "auxiliary_mean_proposal_score": 0.0,
                "auxiliary_proposals": [],
            },
            {
                "family": "alpha",
                "expected_mode": "conflict",
                "anchor_position_mean_future_influence": 0.5,
                "future_span_overlap_ratio": 0.0,
                "auxiliary_proposal_count": 2,
                "auxiliary_mean_proposal_score": 0.8,
                "auxiliary_proposals": [{"proposal_text": "hint one"}, {"proposal_text": "hint two"}],
            },
            {
                "family": "beta",
                "expected_mode": "stable",
                "anchor_position_mean_future_influence": 0.2,
                "future_span_overlap_ratio": 0.0,
                "auxiliary_proposal_count": 1,
                "auxiliary_mean_proposal_score": 0.6,
                "auxiliary_proposals": [{"proposal_text": "stable"}],
            },
            {
                "family": "beta",
                "expected_mode": "conflict",
                "anchor_position_mean_future_influence": 0.4,
                "future_span_overlap_ratio": 0.1,
                "auxiliary_proposal_count": 1,
                "auxiliary_mean_proposal_score": 0.4,
                "auxiliary_proposals": [{"proposal_text": "conflict"}],
            },
        ]
    }

    report_data = build_auxiliary_report_data(anchor_payload, future_payload)

    assert report_data["summary"]["family_count"] == 2
    assert report_data["summary"]["count_wins"] == 1
    assert report_data["summary"]["score_wins"] == 1
    assert report_data["summary"]["future_rescue_count"] == 1
    alpha_row = next(row for row in report_data["families"] if row["family"] == "alpha")
    assert alpha_row["classification"] == "future_rescue"
    assert alpha_row["aux_count_gap"] == 2


def test_build_auxiliary_revision_report_data_tracks_revise_gain():
    anchor_payload = {
        "results": [
            {"family": "alpha", "expected_mode": "stable", "mean_contradiction_pressure": 0.4, "mean_viability": 0.4},
            {"family": "alpha", "expected_mode": "conflict", "mean_contradiction_pressure": 0.3, "mean_viability": 0.5},
            {"family": "beta", "expected_mode": "stable", "mean_contradiction_pressure": 0.2, "mean_viability": 0.7},
            {"family": "beta", "expected_mode": "conflict", "mean_contradiction_pressure": 0.4, "mean_viability": 0.3},
        ]
    }
    future_payload = {
        "results": [
            {
                "family": "alpha",
                "expected_mode": "stable",
                "anchor_position_mean_future_influence": 0.1,
                "future_span_overlap_ratio": 0.0,
                "auxiliary_revision_matched_anchor_count": 0,
                "auxiliary_revision_revise_gain": 0,
                "auxiliary_revision_retire_delta": 0,
                "auxiliary_revision_mean_alt_prob": 0.0,
                "auxiliary_proposals": [],
            },
            {
                "family": "alpha",
                "expected_mode": "conflict",
                "anchor_position_mean_future_influence": 0.5,
                "future_span_overlap_ratio": 0.0,
                "auxiliary_revision_matched_anchor_count": 2,
                "auxiliary_revision_revise_gain": 1,
                "auxiliary_revision_retire_delta": -1,
                "auxiliary_revision_mean_alt_prob": 0.7,
                "auxiliary_proposals": [{"proposal_text": "hint one"}],
            },
            {
                "family": "beta",
                "expected_mode": "stable",
                "anchor_position_mean_future_influence": 0.2,
                "future_span_overlap_ratio": 0.0,
                "auxiliary_revision_matched_anchor_count": 1,
                "auxiliary_revision_revise_gain": 0,
                "auxiliary_revision_retire_delta": 0,
                "auxiliary_revision_mean_alt_prob": 0.3,
                "auxiliary_proposals": [{"proposal_text": "stable"}],
            },
            {
                "family": "beta",
                "expected_mode": "conflict",
                "anchor_position_mean_future_influence": 0.4,
                "future_span_overlap_ratio": 0.1,
                "auxiliary_revision_matched_anchor_count": 1,
                "auxiliary_revision_revise_gain": 0,
                "auxiliary_revision_retire_delta": 1,
                "auxiliary_revision_mean_alt_prob": 0.2,
                "auxiliary_proposals": [{"proposal_text": "conflict"}],
            },
        ]
    }

    report_data = build_auxiliary_revision_report_data(anchor_payload, future_payload)

    assert report_data["summary"]["family_count"] == 2
    assert report_data["summary"]["match_wins"] == 1
    assert report_data["summary"]["revise_gain_wins"] == 1
    alpha_row = next(row for row in report_data["families"] if row["family"] == "alpha")
    assert alpha_row["classification"] == "future_rescue"
    assert alpha_row["revise_gain_gap"] == 1


def test_qwen_overlay_emits_future_hint_batches():
    model = _DummyCausalLM()
    tokenizer = _DummyTokenizer()
    overlay = QwenAnchorOverlay(base_model=model, cfg=TOY_CONFIG, tokenizer=tokenizer)

    out, _ = overlay.analyze_texts_with_future_influence(
        ["alpha beta"],
        max_length=16,
        future_window=4,
        span_threshold=0.5,
        top_spans=3,
    )

    assert "future_hint_batches" in out
    assert len(out["future_hint_batches"]) == 1
    assert "future_hint_candidates" in out["future_hint_batches"][0]
    assert "auxiliary_proposal_batches" in out
    assert "auxiliary_proposal_diagnostics" in out
    assert "auxiliary_revision_diagnostics" in out


def test_auxiliary_arbiter_uses_one_to_one_matching():
    model = _DummyCausalLM()
    tokenizer = _DummyTokenizer()
    overlay = QwenAnchorOverlay(base_model=model, cfg=TOY_CONFIG, tokenizer=tokenizer)
    anchors = [
        [
            AnchorRecord(
                id=1,
                start_idx=0,
                end_idx=1,
                repr=torch.ones(4),
                score=0.8,
                state=AnchorState.CANDIDATE,
                support=0.8,
                contradiction_pressure=0.8,
                viability=0.2,
                ttl=4.0,
                descendant_mass=0.0,
                descendant_coherence=0.0,
            ),
            AnchorRecord(
                id=2,
                start_idx=2,
                end_idx=3,
                repr=torch.ones(4),
                score=0.8,
                state=AnchorState.CANDIDATE,
                support=0.8,
                contradiction_pressure=0.8,
                viability=0.2,
                ttl=4.0,
                descendant_mass=0.0,
                descendant_coherence=0.0,
            ),
        ]
    ]
    proposals = [[
        {
            "proposal_type": "future_hint_span",
            "proposal_score": 0.95,
            "proposal_span": (6, 7),
            "proposal_root_token": 42,
            "proposal_text": "hint",
        }
    ]]

    arbiter, summaries = overlay._build_auxiliary_arbiter(anchors, proposals)

    assert len(arbiter) == 1
    assert summaries[0]["matched_anchor_count"] == 1
