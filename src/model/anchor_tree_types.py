from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class AnchorTreeRole(str, Enum):
    CONSTRAINT = "constraint"
    STEP = "step"
    DERIVED = "derived"
    REPAIR = "repair"
    DRIFT = "drift"
    ALTERNATIVE = "alternative"
    META = "meta"


class AnchorTreeRelation(str, Enum):
    CHILD = "child"
    SUPPORTS = "supports"
    REQUIRES = "requires"
    COMPATIBLE = "compatible"
    CONTRADICTS = "contradicts"
    EXPECTED_NEXT = "expected_next"
    ALTERNATIVE_TO = "alternative_to"


@dataclass
class AnchorTreeNode:
    node_id: str
    label: str
    text: str
    depth: int
    role: AnchorTreeRole
    source: str
    anchor_id: int | None = None
    span_start: int | None = None
    span_end: int | None = None
    repr: torch.Tensor | None = None
    score: float = 0.0
    required: bool = False
    drift_flag: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnchorTreeEdge:
    parent_id: str
    child_id: str
    relation: AnchorTreeRelation = AnchorTreeRelation.CHILD
    score: float = 1.0


@dataclass
class AnchorTree:
    tree_id: str
    root_id: str
    nodes: dict[str, AnchorTreeNode]
    edges: list[AnchorTreeEdge]
    domain: str
    source_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def root(self) -> AnchorTreeNode:
        return self.nodes[self.root_id]

    def children_of(self, node_id: str) -> list[AnchorTreeNode]:
        children = [self.nodes[edge.child_id] for edge in self.edges if edge.parent_id == node_id]
        children.sort(key=lambda item: (item.depth, item.node_id))
        return children

    def required_node_ids(self) -> list[str]:
        return [node_id for node_id, node in self.nodes.items() if node.required]

    def validate(self) -> None:
        if self.root_id not in self.nodes:
            raise ValueError(f"Unknown root node: {self.root_id}")
        for edge in self.edges:
            if edge.parent_id not in self.nodes:
                raise ValueError(f"Unknown edge parent: {edge.parent_id}")
            if edge.child_id not in self.nodes:
                raise ValueError(f"Unknown edge child: {edge.child_id}")


@dataclass
class TreeMatchPair:
    observed_id: str
    expected_id: str
    score: float


@dataclass
class TreeMatch:
    observed_tree_id: str
    expected_tree_id: str
    pairs: list[TreeMatchPair]
    unmatched_observed_ids: list[str]
    unmatched_expected_ids: list[str]
    missing_required_ids: list[str]
    coverage: float
    spurious_ratio: float
    alignment_score: float
    order_violations: int = 0


@dataclass
class TreeConsistencyDiagnostics:
    tree_id: str
    domain: str
    coverage: float
    spurious_ratio: float
    alignment_score: float
    missing_required_count: int
    order_violations: int
    drift_score: float


@dataclass
class ProposalRepairDiagnostics:
    proposal_node_id: str
    proposal_label: str
    repair_gain: float
    coverage_delta: float
    spurious_delta: float
    alignment_delta: float
    notes: dict[str, Any] = field(default_factory=dict)

