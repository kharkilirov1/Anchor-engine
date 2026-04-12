from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from src.model.anchor_types import AnchorRecord
from src.model.future_influence import FutureInfluenceScorer


@dataclass
class AnchorDependencyEdge:
    source_id: int
    target_id: int
    approx_score: float
    final_score: float
    similarity: float
    temporal_prior: float
    support_prior: float
    viability_prior: float
    refined_delta: float = 0.0
    is_refined: bool = False


@dataclass
class AnchorDependencyNode:
    anchor_id: int
    validity: float
    soft_confirmation: float
    node_pressure: float
    predecessor_ids: list[int]
    broken_predecessor_ids: list[int]


def _to_float(value: torch.Tensor | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _sigmoid_unit(value: float, threshold: float, slope: float) -> float:
    safe_slope = max(float(slope), 1e-6)
    tensor = torch.tensor((float(value) - float(threshold)) / safe_slope, dtype=torch.float32)
    return float(torch.sigmoid(tensor).item())


def _temporal_prior(source: AnchorRecord, target: AnchorRecord, temporal_window: float) -> float:
    distance = max(1, int(target.start_idx) - int(source.end_idx))
    return float(math.exp(-(float(distance) - 1.0) / max(float(temporal_window), 1e-6)))


def _approx_dependency_score(
    source: AnchorRecord,
    target: AnchorRecord,
    *,
    confirm_threshold: float,
    similarity_weight: float,
    temporal_weight: float,
    support_weight: float,
    viability_weight: float,
    temporal_window: float,
) -> tuple[float, dict[str, float]]:
    source_repr = F.normalize(source.repr.detach().float().unsqueeze(0), dim=-1)
    target_repr = F.normalize(target.repr.detach().float().unsqueeze(0), dim=-1)
    similarity = max(0.0, float(F.cosine_similarity(source_repr, target_repr, dim=-1).item()))
    temporal = _temporal_prior(source, target, temporal_window)
    support = min(1.0, max(0.0, 0.5 * (_to_float(source.support) + _to_float(target.support))))
    viability = min(
        1.0,
        max(
            0.0,
            0.5
            * (
                _sigmoid_unit(_to_float(source.support), confirm_threshold, 0.10)
                + _to_float(target.viability)
            ),
        ),
    )
    total_weight = max(
        float(similarity_weight) + float(temporal_weight) + float(support_weight) + float(viability_weight),
        1e-6,
    )
    score = (
        float(similarity_weight) * similarity
        + float(temporal_weight) * temporal
        + float(support_weight) * support
        + float(viability_weight) * viability
    ) / total_weight
    return float(score), {
        "similarity": float(similarity),
        "temporal_prior": float(temporal),
        "support_prior": float(support),
        "viability_prior": float(viability),
    }


def _compute_counterfactual_scores(
    *,
    anchors: list[AnchorRecord],
    candidate_edges: list[AnchorDependencyEdge],
    hidden: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    output_projection: torch.nn.Module | None,
    future_scorer: FutureInfluenceScorer | None,
    future_window: int,
    max_edges: int,
) -> dict[tuple[int, int], float]:
    if not candidate_edges or hidden is None or input_ids is None or output_projection is None or future_scorer is None:
        return {}
    if hidden.ndim != 3 or hidden.size(0) != 1 or input_ids.ndim != 2:
        return {}

    edge_map = {(edge.source_id, edge.target_id): edge for edge in candidate_edges}
    top_edges = sorted(candidate_edges, key=lambda item: item.approx_score, reverse=True)[: max(0, int(max_edges))]
    if not top_edges:
        return {}

    anchor_by_id = {anchor.id: anchor for anchor in anchors}
    base_hidden = hidden.detach().clone().requires_grad_(True)
    base_logits = output_projection(base_hidden)
    base_scores = future_scorer(
        hidden=base_hidden,
        logits=base_logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        future_window=future_window,
    )["scores"].detach()

    deltas: dict[tuple[int, int], float] = {}
    unique_source_ids = sorted({edge.source_id for edge in top_edges})
    for source_id in unique_source_ids:
        source_anchor = anchor_by_id.get(source_id)
        if source_anchor is None:
            continue
        masked_hidden = hidden.detach().clone()
        start = max(0, int(source_anchor.start_idx))
        end = min(masked_hidden.size(1) - 1, int(source_anchor.end_idx))
        masked_hidden[:, start : end + 1, :] = 0.0
        masked_hidden = masked_hidden.requires_grad_(True)
        masked_logits = output_projection(masked_hidden)
        masked_scores = future_scorer(
            hidden=masked_hidden,
            logits=masked_logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            future_window=future_window,
        )["scores"].detach()

        for edge in top_edges:
            if edge.source_id != source_id:
                continue
            target_anchor = anchor_by_id.get(edge.target_id)
            if target_anchor is None:
                continue
            target_start = max(0, int(target_anchor.start_idx))
            target_end = min(base_scores.size(1) - 1, int(target_anchor.end_idx))
            if target_end < target_start:
                continue
            delta = (
                base_scores[:, target_start : target_end + 1] - masked_scores[:, target_start : target_end + 1]
            ).abs().mean()
            deltas[(edge.source_id, edge.target_id)] = float(torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=0.0).item())
    return deltas


def build_anchor_dependency_graph(
    anchors: list[AnchorRecord],
    *,
    confirm_threshold: float,
    dependency_threshold: float = 0.55,
    confirm_slope: float = 0.10,
    similarity_weight: float = 0.55,
    temporal_weight: float = 0.20,
    support_weight: float = 0.15,
    viability_weight: float = 0.10,
    temporal_window: float = 16.0,
    max_predecessors: int = 4,
    counterfactual_top_edges: int = 0,
    future_scorer: FutureInfluenceScorer | None = None,
    hidden: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    output_projection: torch.nn.Module | None = None,
    future_window: int = 16,
) -> dict[str, Any]:
    if not anchors:
        return {
            "edges": [],
            "nodes": [],
            "graph_pressure": 0.0,
            "current_graph_pressure": 0.0,
            "current_anchor_id": None,
            "edge_count": 0,
            "broken_anchor_count": 0,
            "mean_validity": 1.0,
        }

    sorted_anchors = sorted(anchors, key=lambda item: (int(item.start_idx), int(item.end_idx), int(item.id)))
    anchor_by_id = {anchor.id: anchor for anchor in sorted_anchors}
    edges_by_target: dict[int, list[AnchorDependencyEdge]] = {anchor.id: [] for anchor in sorted_anchors}
    candidate_edges: list[AnchorDependencyEdge] = []
    for source in sorted_anchors:
        for target in sorted_anchors:
            if int(source.end_idx) >= int(target.start_idx) or source.id == target.id:
                continue
            approx_score, parts = _approx_dependency_score(
                source,
                target,
                confirm_threshold=confirm_threshold,
                similarity_weight=similarity_weight,
                temporal_weight=temporal_weight,
                support_weight=support_weight,
                viability_weight=viability_weight,
                temporal_window=temporal_window,
            )
            if approx_score < float(dependency_threshold):
                continue
            candidate_edges.append(
                AnchorDependencyEdge(
                    source_id=source.id,
                    target_id=target.id,
                    approx_score=float(approx_score),
                    final_score=float(approx_score),
                    similarity=float(parts["similarity"]),
                    temporal_prior=float(parts["temporal_prior"]),
                    support_prior=float(parts["support_prior"]),
                    viability_prior=float(parts["viability_prior"]),
                )
            )

    deltas = _compute_counterfactual_scores(
        anchors=sorted_anchors,
        candidate_edges=candidate_edges,
        hidden=hidden,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_projection=output_projection,
        future_scorer=future_scorer,
        future_window=future_window,
        max_edges=counterfactual_top_edges,
    )

    for edge in candidate_edges:
        edge_key = (edge.source_id, edge.target_id)
        delta = float(deltas.get(edge_key, 0.0))
        if edge_key in deltas:
            edge.refined_delta = delta
            edge.is_refined = True
            edge.final_score = 0.5 * float(edge.approx_score) + 0.5 * min(1.0, max(0.0, delta))

    for anchor in sorted_anchors:
        incoming = [edge for edge in candidate_edges if edge.target_id == anchor.id]
        incoming.sort(key=lambda item: item.final_score, reverse=True)
        edges_by_target[anchor.id] = incoming[: max(1, int(max_predecessors))] if incoming else []

    nodes: list[AnchorDependencyNode] = []
    node_by_id: dict[int, AnchorDependencyNode] = {}
    for anchor in sorted_anchors:
        soft_confirmation = _sigmoid_unit(_to_float(anchor.support), confirm_threshold, confirm_slope)
        predecessors = edges_by_target[anchor.id]
        if predecessors:
            total = sum(edge.final_score for edge in predecessors)
            weighted_confirmation = sum(
                edge.final_score * _sigmoid_unit(
                    _to_float(anchor_by_id[edge.source_id].support),
                    confirm_threshold,
                    confirm_slope,
                )
                for edge in predecessors
            ) / max(total, 1e-6)
        else:
            weighted_confirmation = 1.0
        broken_predecessors = [
            edge.source_id
            for edge in predecessors
            if _sigmoid_unit(
                _to_float(anchor_by_id[edge.source_id].support),
                confirm_threshold,
                confirm_slope,
            )
            < 0.5
        ]
        node_pressure = 1.0 - float(weighted_confirmation) * min(1.0, max(0.0, _to_float(anchor.viability)))
        node = AnchorDependencyNode(
            anchor_id=anchor.id,
            validity=float(weighted_confirmation),
            soft_confirmation=float(soft_confirmation),
            node_pressure=float(min(1.0, max(0.0, node_pressure))),
            predecessor_ids=[edge.source_id for edge in predecessors],
            broken_predecessor_ids=broken_predecessors,
        )
        nodes.append(node)
        node_by_id[anchor.id] = node

    current_anchor = max(sorted_anchors, key=lambda item: (int(item.end_idx), int(item.start_idx), int(item.id)))
    current_graph_pressure = float(node_by_id[current_anchor.id].node_pressure)
    graph_pressure = max((node.node_pressure for node in nodes), default=0.0)
    return {
        "edges": [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "approx_score": edge.approx_score,
                "final_score": edge.final_score,
                "similarity": edge.similarity,
                "temporal_prior": edge.temporal_prior,
                "support_prior": edge.support_prior,
                "viability_prior": edge.viability_prior,
                "refined_delta": edge.refined_delta,
                "is_refined": edge.is_refined,
            }
            for target_edges in edges_by_target.values()
            for edge in target_edges
        ],
        "nodes": [
            {
                "anchor_id": node.anchor_id,
                "validity": node.validity,
                "soft_confirmation": node.soft_confirmation,
                "node_pressure": node.node_pressure,
                "predecessor_ids": node.predecessor_ids,
                "broken_predecessor_ids": node.broken_predecessor_ids,
            }
            for node in nodes
        ],
        "graph_pressure": float(graph_pressure),
        "current_graph_pressure": float(current_graph_pressure),
        "current_anchor_id": int(current_anchor.id),
        "edge_count": int(sum(len(edges) for edges in edges_by_target.values())),
        "broken_anchor_count": int(sum(1 for node in nodes if node.broken_predecessor_ids)),
        "mean_validity": float(sum(node.validity for node in nodes) / max(len(nodes), 1)),
    }
