from __future__ import annotations

import torch

from src.model.qwen_candidate_rerank import (
    branch_aware_candidate_score,
    candidate_average_logprob,
    compute_anchor_bonus,
    compute_tree_bonus,
    extract_tree_candidate_metrics,
    longest_common_prefix_length,
    reranked_candidate_score,
    select_best_branch,
)


def test_longest_common_prefix_length_stops_on_first_mismatch() -> None:
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert longest_common_prefix_length([1, 2], [3, 4]) == 0


def test_candidate_average_logprob_scores_suffix_only() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
                [1.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    score = candidate_average_logprob(logits=logits, input_ids=input_ids, prefix_length=2)
    expected_log_probs = torch.log_softmax(logits[:, 1:3, :], dim=-1)
    expected = torch.stack(
        [
            expected_log_probs[0, 0, 2],
            expected_log_probs[0, 1, 0],
        ]
    ).mean().item()
    assert abs(score - expected) < 1e-6


def test_compute_anchor_bonus_rewards_healthier_candidate() -> None:
    healthy = compute_anchor_bonus(
        mean_contradiction_pressure=0.2,
        mean_viability=0.8,
        dead_end_count=1,
        auxiliary_revision_revise_gain=1,
        auxiliary_revision_retire_delta=-1,
        auxiliary_revision_matched_anchor_count=2,
    )
    unhealthy = compute_anchor_bonus(
        mean_contradiction_pressure=0.8,
        mean_viability=0.2,
        dead_end_count=4,
        auxiliary_revision_revise_gain=0,
        auxiliary_revision_retire_delta=1,
        auxiliary_revision_matched_anchor_count=0,
    )
    assert healthy > unhealthy


def test_reranked_candidate_score_adds_anchor_bonus() -> None:
    score = reranked_candidate_score(base_average_logprob=-2.0, anchor_bonus=0.5, rerank_strength=0.4)
    assert abs(score - (-1.8)) < 1e-6


def test_compute_tree_bonus_rewards_structurally_healthier_candidate() -> None:
    healthy = compute_tree_bonus(
        coverage=0.8,
        alignment_score=0.7,
        drift_score=0.2,
        best_repair_gain=0.4,
        graph_consistency_score=0.9,
        mean_repair_gain=0.3,
    )
    unhealthy = compute_tree_bonus(
        coverage=0.2,
        alignment_score=0.1,
        drift_score=1.2,
        best_repair_gain=-0.2,
        graph_consistency_score=0.5,
        mean_repair_gain=-0.1,
    )
    assert healthy > unhealthy


def test_branch_aware_candidate_score_adds_tree_bonus() -> None:
    score = branch_aware_candidate_score(
        base_average_logprob=-2.0,
        anchor_bonus=0.5,
        tree_bonus=0.4,
        rerank_strength=0.4,
        tree_strength=0.5,
    )
    assert abs(score - (-1.6)) < 1e-6


def test_extract_tree_candidate_metrics_reads_overlay_tree_outputs() -> None:
    out = {
        "observed_tree_batches": [
            {
                "domain": "math_ibp",
                "tree_diagnostics": {
                    "coverage": 0.75,
                    "alignment_score": 0.6,
                    "spurious_ratio": 0.1,
                    "drift_score": 0.2,
                },
                "proposal_repair": [
                    type("Repair", (), {"repair_gain": 0.35})(),
                ],
            }
        ],
        "observed_tree_graph_diagnostics": {
            "graph_consistency_score": 0.8,
            "mean_pair_conflict": 0.1,
        },
        "auxiliary_revision_diagnostics": {
            "mean_repair_gain": 0.25,
        },
    }

    metrics = extract_tree_candidate_metrics(out)

    assert metrics["tree_domain"] == "math_ibp"
    assert metrics["tree_best_repair_gain"] == 0.35
    assert metrics["tree_bonus"] > 0


def test_select_best_branch_uses_requested_score_key() -> None:
    candidates = [
        {"name": "a", "branch_score": 0.1},
        {"name": "b", "branch_score": 0.4},
    ]

    best = select_best_branch(candidates, score_key="branch_score")

    assert best["name"] == "b"
