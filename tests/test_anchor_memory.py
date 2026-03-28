from dataclasses import replace

import torch

from src.model.anchor_memory import AnchorMemory
from src.model.anchor_types import AnchorCandidate, AnchorState, RevisionDecision
from src.model.config import TOY_CONFIG


def test_anchor_memory_add_and_export():
    cfg = replace(TOY_CONFIG)
    memory = AnchorMemory(cfg)
    candidates = [[
        AnchorCandidate(
            start_idx=0,
            end_idx=1,
            repr=torch.randn(cfg.d_model),
            score=0.8,
            semantic_weight=1.2,
        )
    ]]

    anchors = memory.add_candidates(candidates)
    diagnostics = memory.export_diagnostics(anchors)

    assert len(anchors) == 1
    assert len(anchors[0]) == 1
    assert diagnostics["num_active"] == 1
    assert diagnostics["state_counts"][AnchorState.CANDIDATE.value] == 1


def test_anchor_memory_apply_revision():
    cfg = replace(TOY_CONFIG)
    memory = AnchorMemory(cfg)
    candidates = [[
        AnchorCandidate(
            start_idx=0,
            end_idx=1,
            repr=torch.randn(cfg.d_model),
            score=0.8,
            semantic_weight=1.2,
        )
    ]]
    anchors = memory.add_candidates(candidates)
    anchor_id = anchors[0][0].id

    decisions = [
        RevisionDecision(
            anchor_id=anchor_id,
            action="retire",
            reason="test",
            new_state=AnchorState.DEAD_END,
            alt_branch_used=False,
        )
    ]
    anchors = memory.apply_revision(anchors, decisions)

    assert anchors[0][0].state == AnchorState.DEAD_END
