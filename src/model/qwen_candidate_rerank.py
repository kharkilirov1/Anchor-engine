from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def longest_common_prefix_length(a: list[int], b: list[int]) -> int:
    length = 0
    for left, right in zip(a, b):
        if left != right:
            break
        length += 1
    return length


def candidate_average_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prefix_length: int,
) -> float:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, seq, vocab] and input_ids must be [batch, seq]")
    if logits.size(0) != 1 or input_ids.size(0) != 1:
        raise ValueError("candidate_average_logprob expects batch size 1")

    seq_len = int(input_ids.size(1))
    if seq_len < 2 or prefix_length >= seq_len:
        return float("-inf")

    start = max(1, int(prefix_length))
    if start >= seq_len:
        return float("-inf")

    target_ids = input_ids[:, start:]
    target_logits = logits[:, start - 1 : seq_len - 1, :]
    log_probs = F.log_softmax(target_logits, dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().item())


def compute_anchor_bonus(
    mean_contradiction_pressure: float,
    mean_viability: float,
    dead_end_count: int,
    auxiliary_revision_revise_gain: int,
    auxiliary_revision_retire_delta: int,
    auxiliary_revision_matched_anchor_count: int,
) -> float:
    return (
        0.90 * float(mean_viability)
        - 0.90 * float(mean_contradiction_pressure)
        - 0.08 * float(dead_end_count)
        + 0.12 * float(auxiliary_revision_revise_gain)
        - 0.06 * float(auxiliary_revision_retire_delta)
        + 0.04 * float(auxiliary_revision_matched_anchor_count)
    )


def compute_tree_bonus(
    coverage: float,
    alignment_score: float,
    drift_score: float,
    best_repair_gain: float,
    graph_consistency_score: float,
    mean_repair_gain: float,
) -> float:
    return (
        0.60 * float(coverage)
        + 0.55 * float(alignment_score)
        - 0.45 * float(drift_score)
        + 0.35 * float(best_repair_gain)
        + 0.20 * float(graph_consistency_score)
        + 0.20 * float(mean_repair_gain)
    )


def reranked_candidate_score(
    base_average_logprob: float,
    anchor_bonus: float,
    rerank_strength: float,
) -> float:
    return float(base_average_logprob) + float(rerank_strength) * float(anchor_bonus)


def branch_aware_candidate_score(
    base_average_logprob: float,
    anchor_bonus: float,
    tree_bonus: float,
    rerank_strength: float,
    tree_strength: float,
) -> float:
    return (
        float(base_average_logprob)
        + float(rerank_strength) * float(anchor_bonus)
        + float(tree_strength) * float(tree_bonus)
    )


def extract_candidate_metrics(
    out: dict[str, Any],
) -> dict[str, float]:
    diag = out["anchor_diagnostics"]
    aux = out["auxiliary_revision_diagnostics"]
    anchor_bonus = compute_anchor_bonus(
        mean_contradiction_pressure=float(diag["mean_contradiction_pressure"]),
        mean_viability=float(diag["mean_viability"]),
        dead_end_count=int(diag["dead_end_count"]),
        auxiliary_revision_revise_gain=int(aux["auxiliary_revise_gain"]),
        auxiliary_revision_retire_delta=int(aux["auxiliary_retire_delta"]),
        auxiliary_revision_matched_anchor_count=int(aux["matched_anchor_count"]),
    )
    return {
        "mean_contradiction_pressure": float(diag["mean_contradiction_pressure"]),
        "mean_viability": float(diag["mean_viability"]),
        "dead_end_count": int(diag["dead_end_count"]),
        "auxiliary_revision_revise_gain": int(aux["auxiliary_revise_gain"]),
        "auxiliary_revision_retire_delta": int(aux["auxiliary_retire_delta"]),
        "auxiliary_revision_matched_anchor_count": int(aux["matched_anchor_count"]),
        "auxiliary_revision_mean_alt_prob": float(aux["mean_alt_prob"]),
        "auxiliary_revision_mean_repair_gain": float(aux.get("mean_repair_gain", 0.0)),
        "anchor_bonus": float(anchor_bonus),
    }


def extract_tree_candidate_metrics(
    out: dict[str, Any],
) -> dict[str, float]:
    observed_batches = out.get("observed_tree_batches") or []
    graph_diag = out.get("observed_tree_graph_diagnostics") or {}
    batch_diag = observed_batches[0] if observed_batches else {}
    tree_diag = batch_diag.get("tree_diagnostics") or {}
    proposal_repair = batch_diag.get("proposal_repair") or []
    tree_domain = str(batch_diag.get("domain", "unknown"))
    best_repair_gain = float(proposal_repair[0].repair_gain) if proposal_repair else 0.0
    graph_consistency_score = float(graph_diag.get("graph_consistency_score", 0.0))
    mean_repair_gain = float(out.get("auxiliary_revision_diagnostics", {}).get("mean_repair_gain", 0.0))
    if tree_domain == "unknown":
        tree_bonus = 0.0
    else:
        tree_bonus = compute_tree_bonus(
            coverage=float(tree_diag.get("coverage", 0.0)),
            alignment_score=float(tree_diag.get("alignment_score", 0.0)),
            drift_score=float(tree_diag.get("drift_score", 0.0)),
            best_repair_gain=best_repair_gain,
            graph_consistency_score=graph_consistency_score,
            mean_repair_gain=mean_repair_gain,
        )
    return {
        "tree_domain": tree_domain,
        "tree_coverage": float(tree_diag.get("coverage", 0.0)),
        "tree_alignment_score": float(tree_diag.get("alignment_score", 0.0)),
        "tree_spurious_ratio": float(tree_diag.get("spurious_ratio", 0.0)),
        "tree_drift_score": float(tree_diag.get("drift_score", 0.0)),
        "tree_best_repair_gain": float(best_repair_gain),
        "tree_graph_consistency_score": graph_consistency_score,
        "tree_mean_pair_conflict": float(graph_diag.get("mean_pair_conflict", 0.0)),
        "tree_bonus": float(tree_bonus),
    }


def select_best_branch(candidates: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    return max(candidates, key=lambda item: float(item[score_key]))

