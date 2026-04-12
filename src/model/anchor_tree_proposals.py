from __future__ import annotations

from src.model.anchor_tree import attach_child_node, clone_tree
from src.model.anchor_tree_builder import classify_observed_label
from src.model.anchor_tree_match import greedy_tree_match
from src.model.anchor_tree_types import AnchorTree, AnchorTreeNode, AnchorTreeRelation, AnchorTreeRole, ProposalRepairDiagnostics


def _proposal_role(label: str) -> AnchorTreeRole:
    if label in {
        "shortcut_lookup",
        "table_reference",
        "substitution_switch",
        "wrong_symbolic_step",
        "django_view_reframe",
        "synchronous_handler_reframe",
        "template_rendering_branch",
    }:
        return AnchorTreeRole.DRIFT
    if label == "meta_abort":
        return AnchorTreeRole.META
    return AnchorTreeRole.REPAIR


def rank_proposals_by_tree_repair(
    *,
    current_tree: AnchorTree,
    expected_tree: AnchorTree,
    proposal_candidates: list[dict[str, object]],
) -> list[ProposalRepairDiagnostics]:
    baseline = greedy_tree_match(current_tree, expected_tree)
    base_alignment = float(baseline.alignment_score)
    base_coverage = float(baseline.coverage)
    base_spurious = float(baseline.spurious_ratio)
    parent_id = current_tree.root_id
    non_drift_nodes = [node for node in current_tree.nodes.values() if not node.drift_flag and node.node_id != current_tree.root_id]
    if non_drift_nodes:
        parent_id = max(non_drift_nodes, key=lambda node: (node.depth, node.node_id)).node_id

    diagnostics: list[ProposalRepairDiagnostics] = []
    for idx, proposal in enumerate(proposal_candidates):
        proposal_text = str(proposal.get("proposal_text", "")).strip()
        if not proposal_text:
            continue
        start, end = proposal.get("proposal_span", (0, 0))
        label = classify_observed_label(current_tree.domain, proposal_text)
        node = AnchorTreeNode(
            node_id=f"repair_{idx}",
            label=label,
            text=proposal_text,
            depth=max(node.depth for node in current_tree.nodes.values()) + 1,
            role=_proposal_role(label),
            source="auxiliary_proposal",
            span_start=int(start),
            span_end=int(end),
            repr=proposal.get("repr"),
            score=float(proposal.get("proposal_score", 0.0)),
            drift_flag=_proposal_role(label) in {AnchorTreeRole.DRIFT, AnchorTreeRole.META},
        )
        candidate_tree = attach_child_node(
            clone_tree(current_tree),
            parent_id=parent_id,
            node=node,
            relation=AnchorTreeRelation.ALTERNATIVE_TO if node.drift_flag else AnchorTreeRelation.EXPECTED_NEXT,
            score=float(node.score),
        )
        candidate_match = greedy_tree_match(candidate_tree, expected_tree)
        alignment_delta = float(candidate_match.alignment_score) - base_alignment
        coverage_delta = float(candidate_match.coverage) - base_coverage
        spurious_delta = float(candidate_match.spurious_ratio) - base_spurious
        repair_gain = alignment_delta + coverage_delta - spurious_delta
        diagnostics.append(
            ProposalRepairDiagnostics(
                proposal_node_id=node.node_id,
                proposal_label=node.label,
                repair_gain=float(repair_gain),
                coverage_delta=float(coverage_delta),
                spurious_delta=float(spurious_delta),
                alignment_delta=float(alignment_delta),
                notes={
                    "proposal_text": proposal_text,
                    "baseline_alignment": base_alignment,
                    "candidate_alignment": float(candidate_match.alignment_score),
                },
            )
        )
    diagnostics.sort(key=lambda item: item.repair_gain, reverse=True)
    return diagnostics

