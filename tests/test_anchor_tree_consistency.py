from __future__ import annotations

from src.model.anchor_tree import attach_child_node, clone_tree
from src.model.anchor_tree_consistency import compute_cross_tree_conflict, compute_graph_consistency, compute_tree_consistency
from src.model.anchor_tree_templates import get_expected_tree_template
from src.model.anchor_tree_types import AnchorTreeNode, AnchorTreeRelation, AnchorTreeRole


def test_compute_tree_consistency_penalizes_missing_required_nodes() -> None:
    expected = get_expected_tree_template("math_ibp")
    observed = clone_tree(expected)
    observed.nodes.pop("ibp_constant")
    observed.edges = [edge for edge in observed.edges if edge.child_id != "ibp_constant" and edge.parent_id != "ibp_constant"]

    diagnostics = compute_tree_consistency(observed, expected)

    assert diagnostics.missing_required_count > 0
    assert diagnostics.drift_score > 0


def test_cross_tree_conflict_detects_fastapi_vs_django() -> None:
    fastapi_tree = get_expected_tree_template("code_fastapi")
    django_tree = attach_child_node(
        clone_tree(fastapi_tree),
        parent_id="fastapi_handlers",
        node=AnchorTreeNode(
            node_id="django_branch",
            label="django_view_reframe",
            text="switch to django",
            depth=4,
            role=AnchorTreeRole.DRIFT,
            source="test",
            drift_flag=True,
        ),
        relation=AnchorTreeRelation.ALTERNATIVE_TO,
    )

    conflict = compute_cross_tree_conflict(fastapi_tree, django_tree)

    assert conflict > 0
    graph = compute_graph_consistency([fastapi_tree, django_tree])
    assert graph["mean_pair_conflict"] > 0

