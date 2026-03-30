from __future__ import annotations

import pytest

from src.model.anchor_tree_types import AnchorTree, AnchorTreeEdge, AnchorTreeNode, AnchorTreeRelation, AnchorTreeRole


def test_anchor_tree_validate_rejects_unknown_child() -> None:
    root = AnchorTreeNode(
        node_id="root",
        label="root",
        text="root",
        depth=0,
        role=AnchorTreeRole.CONSTRAINT,
        source="test",
    )
    tree = AnchorTree(
        tree_id="tree",
        root_id="root",
        nodes={"root": root},
        edges=[AnchorTreeEdge(parent_id="root", child_id="missing", relation=AnchorTreeRelation.CHILD)],
        domain="toy",
        source_kind="test",
    )

    with pytest.raises(ValueError):
        tree.validate()


def test_anchor_tree_children_of_returns_sorted_children() -> None:
    root = AnchorTreeNode(
        node_id="root",
        label="root",
        text="root",
        depth=0,
        role=AnchorTreeRole.CONSTRAINT,
        source="test",
    )
    child_a = AnchorTreeNode(
        node_id="b_child",
        label="b",
        text="b",
        depth=2,
        role=AnchorTreeRole.STEP,
        source="test",
    )
    child_b = AnchorTreeNode(
        node_id="a_child",
        label="a",
        text="a",
        depth=1,
        role=AnchorTreeRole.STEP,
        source="test",
    )
    tree = AnchorTree(
        tree_id="tree",
        root_id="root",
        nodes={"root": root, "b_child": child_a, "a_child": child_b},
        edges=[
            AnchorTreeEdge(parent_id="root", child_id="b_child"),
            AnchorTreeEdge(parent_id="root", child_id="a_child"),
        ],
        domain="toy",
        source_kind="test",
    )

    tree.validate()
    assert [node.node_id for node in tree.children_of("root")] == ["a_child", "b_child"]

