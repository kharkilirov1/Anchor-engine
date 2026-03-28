from dataclasses import replace

import torch

from src.data.anchor_semantic_cases import make_semantic_anchor_cases
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.model.anchor_types import AnchorRecord, AnchorState, RevisionDecision
from src.model.config import TOY_CONFIG


def test_abpt_anchor_v1_forward():
    cfg = replace(TOY_CONFIG, anchor_threshold=0.2)
    model = ABPTAnchorV1(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    targets = torch.randint(0, cfg.vocab_size, (2, 24))

    out = model(x, targets)

    assert out["logits"].shape == (2, 24, cfg.vocab_size)
    assert "loss" in out
    assert "anchor_candidates" in out
    assert "anchor_diagnostics" in out
    assert "revision_events" in out
    assert "proposal_diagnostics" in out
    assert "mean_strong_retire_gap" in out["proposal_diagnostics"]
    assert "component_losses" in out
    assert "detector_alignment_loss" in out["component_losses"]
    assert "context_stability_loss" in out["component_losses"]


def test_abpt_anchor_v1_backward():
    cfg = replace(TOY_CONFIG, anchor_threshold=0.2)
    model = ABPTAnchorV1(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    targets = torch.randint(0, cfg.vocab_size, (2, 24))

    out = model(x, targets)
    out["loss"].backward()

    grad_params = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grad_params) > 0


def test_anchor_context_uses_only_active_anchors():
    cfg = replace(TOY_CONFIG, d_model=4)
    model = ABPTAnchorV1(cfg)
    hidden = torch.zeros(1, 4, cfg.d_model)
    hidden[0, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    hidden[0, 1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    hidden[0, 2] = torch.tensor([0.0, 0.0, 3.0, 0.0])
    hidden[0, 3] = torch.tensor([0.0, 0.0, 0.0, 4.0])
    scores = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    active_anchors = [[
        AnchorRecord(
            id=0,
            start_idx=1,
            end_idx=2,
            repr=torch.zeros(cfg.d_model),
            score=1.0,
            state=AnchorState.PROVISIONAL,
            support=1.0,
            contradiction_pressure=0.0,
            viability=1.0,
            ttl=1.0,
        )
    ]]

    context, diagnostics = model._build_anchor_context(hidden, scores, active_anchors)
    expected = torch.tensor([0.0, 1.0, 1.5, 0.0])
    assert torch.allclose(context[0, 0], expected)
    assert diagnostics["anchors_with_proposal_influence"] == 0


def test_alternative_reading_baseline_contract():
    cfg = replace(TOY_CONFIG)
    model = ABPTAnchorV1(cfg)
    seq_hidden = torch.randn(8, cfg.d_model)
    seq_ids = torch.zeros(8, dtype=torch.long)
    anchor = AnchorRecord(
        id=0,
        start_idx=3,
        end_idx=4,
        repr=torch.randn(cfg.d_model),
        score=0.8,
        state=AnchorState.CANDIDATE,
        support=0.8,
        contradiction_pressure=0.2,
        viability=0.7,
        ttl=2.0,
    )

    proposal = model._propose_alternative_reading(seq_hidden, seq_ids, anchor)

    assert "repr" in proposal
    assert "proposal_type" in proposal
    assert proposal["repr"].shape == (cfg.d_model,)
    assert proposal["proposal_type"] == "start_state_baseline"


def test_alternative_reading_can_find_regime_shift_window():
    cfg = replace(TOY_CONFIG)
    model = ABPTAnchorV1(cfg)
    semantic_cases = {case.name: case for case in make_semantic_anchor_cases()}
    case = semantic_cases["forall_exists_conflict"]

    seq_hidden = torch.randn(case.input_ids.size(0), cfg.d_model)
    anchor = AnchorRecord(
        id=1,
        start_idx=0,
        end_idx=1,
        repr=seq_hidden[0],
        score=0.9,
        state=AnchorState.PROVISIONAL,
        support=0.9,
        contradiction_pressure=0.9,
        viability=0.4,
        ttl=4.0,
    )

    proposal = model._propose_alternative_reading(seq_hidden, case.input_ids, anchor)

    assert proposal["proposal_type"] == "regime_shift_window"
    assert proposal["repr"].shape == (cfg.d_model,)
    assert proposal["proposal_score"] > 0.38
    assert proposal["proposal_span"][0] >= 12
    assert proposal["proposal_root_token"] == 12


def test_resolve_anchor_regime_root_uses_descendant_alias():
    seq_ids = torch.tensor([15, 21, 14, 22, 53, 15], dtype=torch.long)

    contradiction_root = ABPTAnchorV1._resolve_anchor_regime_root(seq_ids, 0, 1)
    formal_root = ABPTAnchorV1._resolve_anchor_regime_root(seq_ids, 4, 5)

    assert contradiction_root == 21
    assert formal_root == 51


def test_alternative_reading_can_activate_on_proof_mode_flip():
    cfg = replace(TOY_CONFIG)
    model = ABPTAnchorV1(cfg)
    semantic_cases = {case.name: case for case in make_semantic_anchor_cases()}
    case = semantic_cases["contradiction_direct_conflict"]

    seq_hidden = torch.randn(case.input_ids.size(0), cfg.d_model)
    anchor = AnchorRecord(
        id=3,
        start_idx=3,
        end_idx=4,
        repr=seq_hidden[3],
        score=0.8,
        state=AnchorState.PROVISIONAL,
        support=0.8,
        contradiction_pressure=0.8,
        viability=0.5,
        ttl=4.0,
    )

    proposal = model._propose_alternative_reading(seq_hidden, case.input_ids, anchor)

    assert proposal["proposal_type"] == "regime_shift_window"
    assert proposal["proposal_root_token"] == 24


def test_proposal_gate_calibration_boosts_proof_mode_family():
    gate_plain = 0.5 * 0.5 * 0.3
    gate_boosted = ABPTAnchorV1._calibrate_proposal_gate(
        {"proposal_score": 0.3, "proposal_root_token": 24},
        alt_prob=0.5,
        revise_prob=0.5,
    )

    assert gate_boosted > gate_plain


def test_proposal_root_prior_penalizes_formal_limit_close_replacement():
    assert ABPTAnchorV1._proposal_root_prior(54) < 0.5
    assert ABPTAnchorV1._proposal_root_prior(12) < 1.0
    assert ABPTAnchorV1._proposal_root_prior(15) < 0.5
    assert ABPTAnchorV1._proposal_root_prior(12) > ABPTAnchorV1._proposal_root_prior(54)
    assert ABPTAnchorV1._proposal_root_prior(45) >= ABPTAnchorV1._proposal_root_prior(54)


def test_anchor_local_proposal_prior_penalizes_collapsed_induction_example():
    anchor = AnchorRecord(
        id=99,
        start_idx=0,
        end_idx=1,
        repr=torch.zeros(TOY_CONFIG.d_model),
        score=0.9,
        state=AnchorState.CANDIDATE,
        support=0.8,
        contradiction_pressure=0.9,
        viability=0.3,
        ttl=1.0,
        descendant_coherence=0.0,
    )

    assert ABPTAnchorV1._anchor_local_proposal_prior(anchor, 45) < 0.5
    assert ABPTAnchorV1._anchor_local_proposal_prior(anchor, 12) == 1.0


def test_alternative_reading_keeps_baseline_for_stable_regime():
    cfg = replace(TOY_CONFIG)
    model = ABPTAnchorV1(cfg)
    semantic_cases = {case.name: case for case in make_semantic_anchor_cases()}
    case = semantic_cases["forall_stable"]

    seq_hidden = torch.randn(case.input_ids.size(0), cfg.d_model)
    anchor = AnchorRecord(
        id=2,
        start_idx=0,
        end_idx=1,
        repr=seq_hidden[0],
        score=0.9,
        state=AnchorState.PROVISIONAL,
        support=0.9,
        contradiction_pressure=0.1,
        viability=0.9,
        ttl=4.0,
    )

    proposal = model._propose_alternative_reading(seq_hidden, case.input_ids, anchor)

    assert proposal["proposal_type"] == "start_state_baseline"


def test_anchor_context_can_soft_blend_alternative_proposal():
    cfg = replace(TOY_CONFIG, d_model=4)
    model = ABPTAnchorV1(cfg)
    hidden = torch.zeros(1, 4, cfg.d_model)
    hidden[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden[0, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    hidden[0, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    scores = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    anchor = AnchorRecord(
        id=7,
        start_idx=0,
        end_idx=1,
        repr=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        score=1.0,
        state=AnchorState.PROVISIONAL,
        support=1.0,
        contradiction_pressure=0.9,
        viability=0.5,
        ttl=2.0,
    )
    decision = RevisionDecision(
        anchor_id=7,
        action="revise",
        reason="soft_revise",
        new_state=AnchorState.PROVISIONAL,
        alt_branch_used=True,
        action_probs={"keep": 0.1, "revise": 0.8, "downgrade": 0.05, "retire": 0.05},
    )

    context, diagnostics = model._build_anchor_context(
        hidden,
        scores,
        [[anchor]],
        proposal_map={
            7: {
                "repr": torch.tensor([0.0, 2.0, 0.0, 0.0]),
                "proposal_type": "regime_shift_window",
                "proposal_score": 1.0,
            }
        },
        arbiter_out={7: {"prefer_alt_prob": 1.0}},
        decision_map={7: decision},
    )

    assert context[0, 0, 0] < 1.0
    assert context[0, 0, 1] > 0.0
    assert diagnostics["anchors_with_proposal_influence"] == 1
    assert diagnostics["regime_shift_count"] == 1
