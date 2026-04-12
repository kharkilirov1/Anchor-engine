from __future__ import annotations

import torch

from src.model.anchor_tree_builder import build_observed_tree
from src.model.anchor_tree_proposals import rank_proposals_by_tree_repair
from src.model.anchor_tree_templates import get_expected_tree_template
from src.model.anchor_types import AnchorRecord, AnchorState


def _make_anchor(anchor_id: int, start_idx: int, end_idx: int) -> AnchorRecord:
    return AnchorRecord(
        id=anchor_id,
        start_idx=start_idx,
        end_idx=end_idx,
        repr=torch.ones(8),
        score=0.8,
        state=AnchorState.CONFIRMED,
        support=0.9,
        contradiction_pressure=0.2,
        viability=0.8,
        ttl=3.0,
    )


def test_rank_proposals_by_tree_repair_prefers_math_step_over_shortcut() -> None:
    current_tree = build_observed_tree(
        text="Solve the integral using integration by parts only.",
        active_anchors=[
            {"anchor": _make_anchor(1, 0, 3), "text": "integration by parts only", "start": 0, "end": 3},
            {"anchor": _make_anchor(2, 4, 8), "text": "let u = x^2 and dv = e^x dx", "start": 4, "end": 8},
        ],
        future_hint_candidates=[{"text": "du = 2x dx and v = e^x", "start": 9, "end": 14, "mean_score": 0.7}],
        auxiliary_proposals=[],
    )
    expected_tree = get_expected_tree_template("math_ibp")
    proposals = [
        {"proposal_text": "simplify and add + C", "proposal_span": (15, 18), "proposal_score": 0.8, "repr": torch.ones(8)},
        {"proposal_text": "use a shortcut formula table", "proposal_span": (15, 19), "proposal_score": 0.9, "repr": torch.ones(8)},
    ]

    ranked = rank_proposals_by_tree_repair(
        current_tree=current_tree,
        expected_tree=expected_tree,
        proposal_candidates=proposals,
    )

    assert ranked
    assert ranked[0].proposal_label == "integration_constant"
    assert ranked[0].repair_gain >= ranked[-1].repair_gain

