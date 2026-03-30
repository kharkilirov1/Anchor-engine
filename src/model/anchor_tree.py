from __future__ import annotations

from copy import deepcopy

from src.model.anchor_tree_types import AnchorTree, AnchorTreeEdge, AnchorTreeNode, AnchorTreeRelation


def make_anchor_tree(
    *,
    tree_id: str,
    root: AnchorTreeNode,
    nodes: list[AnchorTreeNode],
    edges: list[AnchorTreeEdge],
    domain: str,
    source_kind: str,
) -> AnchorTree:
    all_nodes = {node.node_id: node for node in [root, *nodes]}
    tree = AnchorTree(
        tree_id=tree_id,
        root_id=root.node_id,
        nodes=all_nodes,
        edges=edges,
        domain=domain,
        source_kind=source_kind,
    )
    tree.validate()
    return tree


def clone_tree(tree: AnchorTree) -> AnchorTree:
    return deepcopy(tree)


def attach_child_node(
    tree: AnchorTree,
    *,
    parent_id: str,
    node: AnchorTreeNode,
    relation: AnchorTreeRelation = AnchorTreeRelation.CHILD,
    score: float = 1.0,
) -> AnchorTree:
    updated = clone_tree(tree)
    updated.nodes[node.node_id] = node
    updated.edges.append(
        AnchorTreeEdge(parent_id=parent_id, child_id=node.node_id, relation=relation, score=score)
    )
    updated.validate()
    return updated

