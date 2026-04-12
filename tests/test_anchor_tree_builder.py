from __future__ import annotations

import torch

from src.model.anchor_tree_builder import build_observed_tree
from src.model.anchor_tree_types import AnchorTreeRole
from src.model.anchor_types import AnchorRecord, AnchorState


def _make_anchor(anchor_id: int, start_idx: int, end_idx: int, *, support: float = 0.9, viability: float = 0.8) -> AnchorRecord:
    return AnchorRecord(
        id=anchor_id,
        start_idx=start_idx,
        end_idx=end_idx,
        repr=torch.ones(8),
        score=0.8,
        state=AnchorState.CONFIRMED,
        support=support,
        contradiction_pressure=0.2,
        viability=viability,
        ttl=3.0,
    )


def test_build_observed_tree_for_math_ibp_uses_domain_root() -> None:
    tree = build_observed_tree(
        text="Solve the integral using integration by parts only.",
        active_anchors=[
            {"anchor": _make_anchor(1, 0, 3), "text": "integration by parts only", "start": 0, "end": 3},
            {"anchor": _make_anchor(2, 4, 8), "text": "let u = x^2 and dv = e^x dx", "start": 4, "end": 8},
        ],
        future_hint_candidates=[
            {"text": "du = 2x dx and v = e^x", "start": 9, "end": 14, "mean_score": 0.7},
            {"text": "simplify and add + C", "start": 15, "end": 18, "mean_score": 0.6},
        ],
        auxiliary_proposals=[],
    )

    assert tree.domain == "math_ibp"
    assert tree.root().label == "integration_by_parts_only"
    labels = {node.label for node in tree.nodes.values()}
    assert "select_u_and_dv" in labels
    assert "derive_du_and_v" in labels
    assert "integration_constant" in labels


def test_build_observed_tree_marks_drift_proposals() -> None:
    tree = build_observed_tree(
        text="Build an async FastAPI endpoint.",
        active_anchors=[
            {"anchor": _make_anchor(1, 0, 2), "text": "async FastAPI service", "start": 0, "end": 2},
        ],
        future_hint_candidates=[],
        auxiliary_proposals=[
            {
                "proposal_text": "switch to a synchronous Django template view",
                "proposal_span": (3, 8),
                "proposal_score": 0.9,
                "repr": torch.ones(8),
            }
        ],
    )

    assert tree.domain == "code_fastapi"
    drift_nodes = [node for node in tree.nodes.values() if node.drift_flag]
    assert drift_nodes
    assert drift_nodes[0].role in {AnchorTreeRole.DRIFT, AnchorTreeRole.META}


def test_build_observed_tree_for_quantifier_recovers_scope_labels() -> None:
    tree = build_observed_tree(
        text="Claim: for all natural numbers n the statement is true.",
        active_anchors=[
            {"anchor": _make_anchor(1, 0, 6), "text": "for all natural numbers", "start": 0, "end": 3},
        ],
        future_hint_candidates=[
            {"text": "reject existential witness drift", "start": 8, "end": 11, "mean_score": 0.8},
            {"text": "restate the universal conclusion", "start": 12, "end": 16, "mean_score": 0.7},
        ],
        auxiliary_proposals=[],
    )

    assert tree.domain == "quantifier"
    assert tree.root().label == "universal_quantifier_scope"
    labels = {node.label for node in tree.nodes.values()}
    assert "preserve_universal_claim" in labels
    assert "reject_existential_drift" in labels
    assert "restate_universal_conclusion" in labels


def test_build_observed_tree_for_proof_mode_marks_direct_switch_as_drift() -> None:
    tree = build_observed_tree(
        text="We assume the negation and derive a contradiction.",
        active_anchors=[
            {"anchor": _make_anchor(1, 0, 6), "text": "assume the negation and derive a contradiction", "start": 0, "end": 6},
        ],
        future_hint_candidates=[
            {"text": "discharge the assumed negation", "start": 8, "end": 11, "mean_score": 0.8},
        ],
        auxiliary_proposals=[
            {
                "proposal_text": "switch to a direct proof from scratch",
                "proposal_span": (12, 17),
                "proposal_score": 0.9,
                "repr": torch.ones(8),
            }
        ],
    )

    assert tree.domain == "proof_mode"
    assert tree.root().label == "proof_by_contradiction_mode"
    drift_nodes = [node for node in tree.nodes.values() if node.label == "direct_proof_switch"]
    assert drift_nodes
    assert drift_nodes[0].role == AnchorTreeRole.DRIFT

