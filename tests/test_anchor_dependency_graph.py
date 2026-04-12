from __future__ import annotations

import torch
import torch.nn as nn

from src.model.anchor_dependency_graph import build_anchor_dependency_graph
from src.model.anchor_types import AnchorRecord, AnchorState


def _make_anchor(
    *,
    anchor_id: int,
    start_idx: int,
    end_idx: int,
    repr_values: list[float],
    support: float,
    viability: float,
) -> AnchorRecord:
    return AnchorRecord(
        id=anchor_id,
        start_idx=start_idx,
        end_idx=end_idx,
        repr=torch.tensor(repr_values, dtype=torch.float32),
        score=support,
        state=AnchorState.CONFIRMED,
        support=support,
        contradiction_pressure=0.0,
        viability=viability,
        ttl=4.0,
        descendant_mass=0.0,
        descendant_coherence=0.0,
    )


def test_dependency_graph_builds_directed_edges_from_earlier_similar_anchors() -> None:
    anchors = [
        _make_anchor(anchor_id=1, start_idx=0, end_idx=1, repr_values=[1.0, 0.0, 0.0], support=0.95, viability=0.90),
        _make_anchor(anchor_id=2, start_idx=4, end_idx=5, repr_values=[0.98, 0.05, 0.0], support=0.90, viability=0.85),
        _make_anchor(anchor_id=3, start_idx=8, end_idx=9, repr_values=[0.0, 1.0, 0.0], support=0.80, viability=0.70),
    ]

    graph = build_anchor_dependency_graph(
        anchors,
        confirm_threshold=0.70,
        dependency_threshold=0.50,
        max_predecessors=2,
    )

    assert graph["edge_count"] >= 1
    assert any(edge["source_id"] == 1 and edge["target_id"] == 2 for edge in graph["edges"])
    assert all(edge["source_id"] != edge["target_id"] for edge in graph["edges"])


def test_dependency_graph_marks_broken_predecessor_and_raises_pressure() -> None:
    anchors = [
        _make_anchor(anchor_id=1, start_idx=0, end_idx=1, repr_values=[1.0, 0.0], support=0.20, viability=0.90),
        _make_anchor(anchor_id=2, start_idx=3, end_idx=4, repr_values=[0.99, 0.01], support=0.92, viability=0.80),
    ]

    graph = build_anchor_dependency_graph(
        anchors,
        confirm_threshold=0.70,
        dependency_threshold=0.40,
        max_predecessors=2,
    )

    target_node = next(node for node in graph["nodes"] if node["anchor_id"] == 2)
    assert 1 in target_node["predecessor_ids"]
    assert 1 in target_node["broken_predecessor_ids"]
    assert target_node["validity"] < 0.5
    assert graph["graph_pressure"] > 0.5


def test_dependency_graph_can_refine_top_edges_with_future_counterfactual() -> None:
    anchors = [
        _make_anchor(anchor_id=1, start_idx=0, end_idx=0, repr_values=[1.0, 0.0, 0.0], support=0.95, viability=0.90),
        _make_anchor(anchor_id=2, start_idx=2, end_idx=2, repr_values=[0.95, 0.05, 0.0], support=0.90, viability=0.85),
    ]
    hidden = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 0.2, 0.0], [0.9, 0.1, 0.0], [0.1, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    projection = nn.Linear(3, 7, bias=False)

    from src.model.future_influence import FutureInfluenceScorer

    graph = build_anchor_dependency_graph(
        anchors,
        confirm_threshold=0.70,
        dependency_threshold=0.40,
        max_predecessors=2,
        counterfactual_top_edges=1,
        future_scorer=FutureInfluenceScorer(),
        hidden=hidden,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_projection=projection,
        future_window=2,
    )

    assert graph["edges"]
    assert any(edge["is_refined"] for edge in graph["edges"])
