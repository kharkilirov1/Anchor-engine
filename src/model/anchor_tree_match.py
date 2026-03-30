from __future__ import annotations

import math
import re
from collections.abc import Iterable

import torch.nn.functional as F

from src.model.anchor_tree_types import AnchorTree, AnchorTreeNode, TreeMatch, TreeMatchPair

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _label_compatibility(observed: AnchorTreeNode, expected: AnchorTreeNode) -> float:
    if _normalize_label(observed.label) == _normalize_label(expected.label):
        return 1.0
    observed_tokens = _tokenize(observed.label)
    expected_tokens = _tokenize(expected.label)
    score = _jaccard(observed_tokens, expected_tokens)
    if score > 0.0:
        return score
    return _jaccard(_tokenize(observed.text), _tokenize(expected.text)) * 0.5


def _role_compatibility(observed: AnchorTreeNode, expected: AnchorTreeNode) -> float:
    if observed.role == expected.role:
        return 1.0
    if {observed.role.value, expected.role.value} <= {"step", "derived"}:
        return 0.7
    if "drift" in {observed.role.value, expected.role.value}:
        return 0.2
    return 0.4


def _depth_compatibility(observed: AnchorTreeNode, expected: AnchorTreeNode) -> float:
    return max(0.0, 1.0 - (abs(observed.depth - expected.depth) / 4.0))


def _repr_similarity(observed: AnchorTreeNode, expected: AnchorTreeNode) -> float:
    if observed.repr is None or expected.repr is None:
        return 0.5
    observed_repr = observed.repr.detach().float().reshape(1, -1)
    expected_repr = expected.repr.detach().float().reshape(1, -1)
    score = F.cosine_similarity(observed_repr, expected_repr).item()
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def compute_node_match_score(observed: AnchorTreeNode, expected: AnchorTreeNode) -> float:
    label_score = _label_compatibility(observed, expected)
    role_score = _role_compatibility(observed, expected)
    depth_score = _depth_compatibility(observed, expected)
    text_score = _jaccard(_tokenize(observed.text), _tokenize(expected.text))
    repr_score = _repr_similarity(observed, expected)
    score = (
        0.35 * label_score
        + 0.20 * role_score
        + 0.15 * depth_score
        + 0.15 * text_score
        + 0.15 * repr_score
    )
    return float(max(0.0, min(1.0, score)))


def _count_order_violations(pairs: list[TreeMatchPair], observed: AnchorTree, expected: AnchorTree) -> int:
    if len(pairs) < 2:
        return 0
    by_expected_depth = sorted(
        pairs,
        key=lambda pair: (
            expected.nodes[pair.expected_id].depth,
            pair.expected_id,
        ),
    )
    violations = 0
    last_depth = -math.inf
    for pair in by_expected_depth:
        observed_depth = observed.nodes[pair.observed_id].depth
        if observed_depth < last_depth:
            violations += 1
        last_depth = max(last_depth, observed_depth)
    return violations


def greedy_tree_match(
    observed: AnchorTree,
    expected: AnchorTree,
    *,
    min_score: float = 0.30,
) -> TreeMatch:
    candidate_pairs: list[tuple[float, str, str]] = []
    for observed_id, observed_node in observed.nodes.items():
        for expected_id, expected_node in expected.nodes.items():
            score = compute_node_match_score(observed_node, expected_node)
            if score >= min_score:
                candidate_pairs.append((score, observed_id, expected_id))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)

    matched_observed: set[str] = set()
    matched_expected: set[str] = set()
    pairs: list[TreeMatchPair] = []

    for score, observed_id, expected_id in candidate_pairs:
        if observed_id in matched_observed or expected_id in matched_expected:
            continue
        matched_observed.add(observed_id)
        matched_expected.add(expected_id)
        pairs.append(TreeMatchPair(observed_id=observed_id, expected_id=expected_id, score=float(score)))

    unmatched_observed_ids = sorted(set(observed.nodes) - matched_observed)
    unmatched_expected_ids = sorted(set(expected.nodes) - matched_expected)
    missing_required_ids = sorted(
        expected_id
        for expected_id in unmatched_expected_ids
        if expected.nodes[expected_id].required
    )

    required_total = max(len(expected.required_node_ids()), 1)
    matched_required = sum(1 for pair in pairs if expected.nodes[pair.expected_id].required)
    coverage = matched_required / required_total
    spurious_ratio = len(unmatched_observed_ids) / max(len(observed.nodes), 1)
    order_violations = _count_order_violations(pairs, observed, expected)
    mean_pair_score = sum(pair.score for pair in pairs) / max(len(pairs), 1)
    alignment_score = mean_pair_score * coverage - 0.40 * spurious_ratio - 0.10 * len(missing_required_ids) - 0.05 * order_violations

    return TreeMatch(
        observed_tree_id=observed.tree_id,
        expected_tree_id=expected.tree_id,
        pairs=pairs,
        unmatched_observed_ids=unmatched_observed_ids,
        unmatched_expected_ids=unmatched_expected_ids,
        missing_required_ids=missing_required_ids,
        coverage=float(max(0.0, min(1.0, coverage))),
        spurious_ratio=float(max(0.0, min(1.0, spurious_ratio))),
        alignment_score=float(alignment_score),
        order_violations=order_violations,
    )


def compute_tree_alignment(match: TreeMatch) -> float:
    return float(match.alignment_score)

