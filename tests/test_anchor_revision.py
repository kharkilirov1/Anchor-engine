from dataclasses import replace

import torch

from src.model.anchor_revision import AnchorArbiter, RevisionController
from src.model.anchor_types import AnchorRecord, AnchorState
from src.model.anchor_viability import ViabilityTracker
from src.model.config import TOY_CONFIG


def test_viability_tracker_outputs():
    cfg = replace(TOY_CONFIG)
    tracker = ViabilityTracker(cfg)
    anchors = [[
        AnchorRecord(
            id=0,
            start_idx=0,
            end_idx=2,
            repr=torch.randn(cfg.d_model),
            score=0.9,
            state=AnchorState.CANDIDATE,
            support=0.9,
            contradiction_pressure=0.1,
            viability=0.0,
            ttl=3.0,
        )
    ]]
    contradiction = {"contradiction_pressure": {0: 0.1}}

    out = tracker(anchors, contradiction)

    assert 0 in out["viability"]
    assert 0 in out["state_updates"]


def test_revision_controller_emits_decision():
    cfg = replace(TOY_CONFIG)
    controller = RevisionController(cfg)
    arbiter = AnchorArbiter(cfg)
    anchor = AnchorRecord(
        id=0,
        start_idx=0,
        end_idx=2,
        repr=torch.randn(cfg.d_model),
        score=0.9,
        state=AnchorState.CONFIRMED,
        support=0.2,
        contradiction_pressure=0.95,
        viability=0.1,
        ttl=1.0,
    )
    arbiter_out = {0: arbiter(torch.randn(8, cfg.d_model), anchor, alt={"repr": torch.randn(cfg.d_model)})}
    viability = {"viability": {0: 0.1}, "state_updates": {0: AnchorState.DEAD_END}}

    decisions = controller([[anchor]], viability, arbiter_out)

    assert len(decisions) == 1
    assert decisions[0].action in {"retire", "revise", "downgrade", "keep"}
    assert decisions[0].action_probs is not None
    assert abs(sum(decisions[0].action_probs.values()) - 1.0) < 1e-6


def test_anchor_arbiter_returns_soft_probability():
    cfg = replace(TOY_CONFIG, anchor_arbiter_beta=8.0)
    arbiter = AnchorArbiter(cfg)
    anchor = AnchorRecord(
        id=2,
        start_idx=0,
        end_idx=1,
        repr=torch.ones(cfg.d_model),
        score=0.8,
        state=AnchorState.CANDIDATE,
        support=0.8,
        contradiction_pressure=0.3,
        viability=0.7,
        ttl=2.0,
    )
    hidden = torch.ones(6, cfg.d_model)
    out = arbiter(hidden, anchor, alt={"repr": -torch.ones(cfg.d_model)})

    assert 0.0 <= out["prefer_current_prob"] <= 1.0
    assert 0.0 <= out["prefer_alt_prob"] <= 1.0
    assert abs((out["prefer_current_prob"] + out["prefer_alt_prob"]) - 1.0) < 1e-6
    assert out["prefer_current_prob"] > 0.5


def test_revision_controller_can_use_arbiter_for_promoted_candidate():
    cfg = replace(
        TOY_CONFIG,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
        anchor_arbiter_revise_threshold=0.45,
    )
    controller = RevisionController(cfg)
    anchor = AnchorRecord(
        id=1,
        start_idx=0,
        end_idx=1,
        repr=torch.randn(cfg.d_model),
        score=0.8,
        state=AnchorState.CANDIDATE,
        support=0.8,
        contradiction_pressure=0.4,
        viability=0.75,
        ttl=2.0,
    )
    arbiter_out = {
        1: {
            "prefer_current": False,
            "prefer_current_prob": 0.2,
            "prefer_alt_prob": 0.8,
            "margin": -0.2,
            "arbiter_score": 0.1,
            "alt_score": 0.3,
        }
    }
    viability = {"viability": {1: 0.75}, "state_updates": {1: AnchorState.PROVISIONAL}}

    decisions = controller([[anchor]], viability, arbiter_out)

    assert len(decisions) == 1
    assert decisions[0].action == "revise"
    assert decisions[0].new_state == AnchorState.PROVISIONAL
    assert decisions[0].action_probs is not None
    assert decisions[0].action_probs["revise"] > decisions[0].action_probs["keep"]


def test_revision_controller_can_override_borderline_induction_retire():
    cfg = replace(
        TOY_CONFIG,
        anchor_revision_temperature=1.0,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    controller = RevisionController(cfg)
    anchor = AnchorRecord(
        id=7,
        start_idx=0,
        end_idx=1,
        repr=torch.randn(cfg.d_model),
        score=0.9,
        state=AnchorState.CONFIRMED,
        support=0.5,
        contradiction_pressure=0.78,
        viability=0.45,
        ttl=1.0,
    )
    viability = {"viability": {7: 0.45}, "state_updates": {7: AnchorState.DEAD_END}}
    arbiter_out = {
        7: {
            "prefer_current": False,
            "prefer_current_prob": 0.55,
            "prefer_alt_prob": 0.45,
            "margin": -0.05,
            "arbiter_score": 0.1,
            "alt_score": 0.15,
            "proposal_score": 0.98,
            "proposal_root_token": 45,
        }
    }
    controller._action_distribution = lambda **kwargs: {  # type: ignore[method-assign]
        "keep": 0.03,
        "revise": 0.41,
        "downgrade": 0.09,
        "retire": 0.47,
    }

    decisions = controller([[anchor]], viability, arbiter_out)

    assert len(decisions) == 1
    assert decisions[0].action == "revise"
    assert decisions[0].reason == "induction_timing_override"
    assert decisions[0].new_state == AnchorState.PROVISIONAL
    assert decisions[0].action_probs["retire"] > decisions[0].action_probs["revise"]


def test_revision_controller_can_override_borderline_complexity_retire():
    cfg = replace(
        TOY_CONFIG,
        anchor_revision_temperature=1.0,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    controller = RevisionController(cfg)
    anchor = AnchorRecord(
        id=8,
        start_idx=0,
        end_idx=1,
        repr=torch.randn(cfg.d_model),
        score=0.9,
        state=AnchorState.CONFIRMED,
        support=0.5,
        contradiction_pressure=0.74,
        viability=0.46,
        ttl=1.0,
    )
    viability = {"viability": {8: 0.46}, "state_updates": {8: AnchorState.DEAD_END}}
    arbiter_out = {
        8: {
            "prefer_current": False,
            "prefer_current_prob": 0.52,
            "prefer_alt_prob": 0.48,
            "margin": -0.03,
            "arbiter_score": 0.1,
            "alt_score": 0.13,
            "proposal_score": 0.90,
            "proposal_root_token": 35,
        }
    }
    controller._action_distribution = lambda **kwargs: {  # type: ignore[method-assign]
        "keep": 0.04,
        "revise": 0.415,
        "downgrade": 0.11,
        "retire": 0.428,
    }

    decisions = controller([[anchor]], viability, arbiter_out)

    assert len(decisions) == 1
    assert decisions[0].action == "revise"
    assert decisions[0].reason == "complexity_timing_override"
    assert decisions[0].new_state == AnchorState.PROVISIONAL
    assert decisions[0].action_probs["retire"] > decisions[0].action_probs["revise"]
