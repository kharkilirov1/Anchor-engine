from __future__ import annotations

from src.model.anchor_tree_match import greedy_tree_match
from src.model.anchor_tree_types import AnchorTree, TreeConsistencyDiagnostics

_CONTRADICTION_LABELS: set[frozenset[str]] = {
    frozenset({"async_fastapi_service", "django_view_reframe"}),
    frozenset({"async_handlers", "synchronous_handler_reframe"}),
    frozenset({"typed_request_models", "template_rendering_branch"}),
    frozenset({"integration_by_parts_only", "shortcut_lookup"}),
    frozenset({"integration_by_parts_only", "table_reference"}),
    frozenset({"derive_du_and_v", "wrong_symbolic_step"}),
}


def compute_tree_consistency(
    tree: AnchorTree,
    expected: AnchorTree | None = None,
) -> TreeConsistencyDiagnostics:
    if expected is not None:
        match = greedy_tree_match(tree, expected)
        missing_required_count = len(match.missing_required_ids)
        drift_score = (
            float(missing_required_count)
            + float(match.spurious_ratio)
            + 0.5 * float(match.order_violations)
        )
        return TreeConsistencyDiagnostics(
            tree_id=tree.tree_id,
            domain=tree.domain,
            coverage=float(match.coverage),
            spurious_ratio=float(match.spurious_ratio),
            alignment_score=float(match.alignment_score),
            missing_required_count=int(missing_required_count),
            order_violations=int(match.order_violations),
            drift_score=float(drift_score),
        )

    drift_nodes = sum(1 for node in tree.nodes.values() if node.drift_flag)
    spurious_ratio = drift_nodes / max(len(tree.nodes), 1)
    return TreeConsistencyDiagnostics(
        tree_id=tree.tree_id,
        domain=tree.domain,
        coverage=0.0,
        spurious_ratio=float(spurious_ratio),
        alignment_score=1.0 - float(spurious_ratio),
        missing_required_count=0,
        order_violations=0,
        drift_score=float(spurious_ratio),
    )


def compute_cross_tree_conflict(tree_a: AnchorTree, tree_b: AnchorTree) -> float:
    labels_a = {node.label for node in tree_a.nodes.values()}
    labels_b = {node.label for node in tree_b.nodes.values()}
    conflict_hits = 0
    for pair in _CONTRADICTION_LABELS:
        if len(pair & labels_a) == 1 and len(pair & labels_b) == 1:
            conflict_hits += 1
    if tree_a.domain != tree_b.domain and tree_a.domain != "unknown" and tree_b.domain != "unknown":
        conflict_hits += 1
    return float(conflict_hits)


def compute_graph_consistency(trees: list[AnchorTree]) -> dict[str, float]:
    if not trees:
        return {
            "tree_count": 0.0,
            "mean_pair_conflict": 0.0,
            "graph_consistency_score": 1.0,
        }
    pair_conflicts: list[float] = []
    for idx, left in enumerate(trees):
        for right in trees[idx + 1 :]:
            pair_conflicts.append(compute_cross_tree_conflict(left, right))
    mean_pair_conflict = sum(pair_conflicts) / max(len(pair_conflicts), 1) if pair_conflicts else 0.0
    graph_consistency_score = 1.0 / (1.0 + mean_pair_conflict)
    return {
        "tree_count": float(len(trees)),
        "mean_pair_conflict": float(mean_pair_conflict),
        "graph_consistency_score": float(graph_consistency_score),
    }

