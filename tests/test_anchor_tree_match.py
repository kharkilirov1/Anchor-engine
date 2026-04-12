from __future__ import annotations

from src.model.anchor_tree import attach_child_node, clone_tree
from src.model.anchor_tree_match import compute_tree_alignment, greedy_tree_match
from src.model.anchor_tree_templates import get_expected_tree_template
from src.model.anchor_tree_types import AnchorTreeNode, AnchorTreeRelation, AnchorTreeRole


def test_tree_match_scores_healthy_tree_above_drift_tree() -> None:
    expected = get_expected_tree_template("math_ibp")
    healthy = clone_tree(expected)

    drift = clone_tree(expected)
    drift.nodes.pop("ibp_derive")
    drift.nodes.pop("ibp_substitute")
    drift.edges = [
        edge
        for edge in drift.edges
        if edge.child_id not in {"ibp_derive", "ibp_substitute"}
        and edge.parent_id not in {"ibp_derive", "ibp_substitute"}
    ]
    drift = attach_child_node(
        drift,
        parent_id="ibp_select",
        node=AnchorTreeNode(
            node_id="ibp_meta",
            label="meta_abort",
            text="This is too hard; maybe use a shortcut.",
            depth=2,
            role=AnchorTreeRole.META,
            source="generated_span",
            drift_flag=True,
        ),
        relation=AnchorTreeRelation.ALTERNATIVE_TO,
    )

    healthy_match = greedy_tree_match(healthy, expected)
    drift_match = greedy_tree_match(drift, expected)

    assert healthy_match.coverage > drift_match.coverage
    assert compute_tree_alignment(healthy_match) > compute_tree_alignment(drift_match)
    assert drift_match.missing_required_ids


def test_tree_match_penalizes_spurious_observed_nodes() -> None:
    expected = get_expected_tree_template("code_fastapi")
    observed = attach_child_node(
        clone_tree(expected),
        parent_id="fastapi_handlers",
        node=AnchorTreeNode(
            node_id="django_branch",
            label="django_view_reframe",
            text="Switch to a synchronous Django class-based view.",
            depth=4,
            role=AnchorTreeRole.DRIFT,
            source="future_hint",
            drift_flag=True,
        ),
        relation=AnchorTreeRelation.ALTERNATIVE_TO,
    )

    match = greedy_tree_match(observed, expected)

    assert match.spurious_ratio > 0.0
    assert match.alignment_score < 1.0

