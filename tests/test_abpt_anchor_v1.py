from dataclasses import replace

import torch

from src.data.anchor_semantic_cases import make_semantic_anchor_cases
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.model.anchor_monitor import ContradictionMonitor
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
    assert "proposal_score_loss" in out["component_losses"]
    assert "proposal_margin_loss" in out["component_losses"]
    assert "proposal_alignment_loss" in out["component_losses"]
    assert "proposal_counterfactual_loss" in out["component_losses"]
    assert "proposal_rollout_loss" in out["component_losses"]
    assert "proposal_aux_metrics" in out
    assert "proposal_counterfactual_gain" in out["proposal_aux_metrics"]
    assert "proposal_rollout_gain" in out["proposal_aux_metrics"]


def test_abpt_anchor_v1_forward_with_fog_flow():
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.2,
        use_fog_flow=True,
        fog_task_profile="stories",
        max_seq_len=24,
    )
    model = ABPTAnchorV1(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    targets = torch.randint(0, cfg.vocab_size, (2, 24))

    out = model(x, targets)

    assert out["logits"].shape == (2, 24, cfg.vocab_size)
    assert out["flow_type"] == "fog_hybrid"
    assert out["fog_profile"] == "stories"
    assert "loss" in out


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

    context, diagnostics, gate_strength = model._build_anchor_context(hidden, scores, active_anchors)
    expected = torch.tensor([0.0, 1.0, 1.5, 0.0])
    assert torch.allclose(context[0, 0], expected)
    assert diagnostics["anchors_with_proposal_influence"] == 0
    assert gate_strength[0, 1, 0] > 0.0
    assert gate_strength[0, 0, 0] == 0.0


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
    cfg = replace(TOY_CONFIG, anchor_domain_mode="synthetic")
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


def test_contradiction_monitor_resolves_descendant_alias_from_span():
    span = torch.tensor([15, 22, 14], dtype=torch.long)

    root = ContradictionMonitor.resolve_regime_root_from_span(span)
    inferred = ContradictionMonitor.infer_reference_root(span)

    assert root == 21
    assert inferred == 21


def test_alternative_reading_can_activate_on_proof_mode_flip():
    cfg = replace(TOY_CONFIG, anchor_domain_mode="synthetic")
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


def test_alternative_reading_can_use_soft_regime_fallback_for_open_vocab():
    cfg = replace(TOY_CONFIG, anchor_domain_mode="synthetic")
    model = ABPTAnchorV1(cfg)
    seq_ids = torch.tensor([70, 71, 70, 72, 80, 81, 80, 81, 80, 81], dtype=torch.long)
    seq_hidden = torch.randn(seq_ids.size(0), cfg.d_model)
    anchor = AnchorRecord(
        id=12,
        start_idx=0,
        end_idx=2,
        repr=seq_hidden[0],
        score=0.8,
        state=AnchorState.PROVISIONAL,
        support=0.8,
        contradiction_pressure=0.9,
        viability=0.4,
        ttl=4.0,
    )

    proposal = model._propose_alternative_reading(seq_hidden, seq_ids, anchor)

    assert proposal["proposal_type"] == "regime_shift_window"
    assert proposal["proposal_root_token"] in {80, 81}
    assert proposal["proposal_score"] > 0.24


def test_anchor_context_can_soft_blend_alternative_proposal():
    cfg = replace(TOY_CONFIG, d_model=4, anchor_domain_mode="synthetic")
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

    context, diagnostics, gate_strength = model._build_anchor_context(
        hidden,
        scores,
        [[anchor]],
        proposal_map={
            7: {
                "repr": torch.tensor([0.0, 2.0, 0.0, 0.0]),
                "branch_repr": torch.tensor([0.0, 1.5, 0.0, 0.0]),
                "proposal_type": "regime_shift_window",
                "proposal_score": 1.0,
                "rollout_steps": 4,
            }
        },
        arbiter_out={7: {"prefer_alt_prob": 1.0}},
        decision_map={7: decision},
    )

    assert context[0, 0, 0] < 1.0
    assert context[0, 0, 1] > 0.0
    assert diagnostics["anchors_with_proposal_influence"] == 1
    assert diagnostics["regime_shift_count"] == 1
    assert diagnostics["rollout_count"] == 1
    assert gate_strength[0, 0, 0] > 0.0


def test_alternative_reading_uses_future_window_head_on_real_domain_mode():
    cfg = replace(TOY_CONFIG, anchor_domain_mode="real")
    model = ABPTAnchorV1(cfg)
    seq_ids = torch.tensor([12, 13, 24, 24, 24, 24, 24, 24], dtype=torch.long)
    seq_hidden = torch.zeros(seq_ids.size(0), cfg.d_model)
    seq_hidden[0:2] = 0.0
    seq_hidden[0:2, 0] = 1.0
    seq_hidden[2:, 1] = 1.0
    anchor = AnchorRecord(
        id=13,
        start_idx=0,
        end_idx=1,
        repr=seq_hidden[0],
        score=0.8,
        state=AnchorState.PROVISIONAL,
        support=0.8,
        contradiction_pressure=0.9,
        viability=0.4,
        ttl=4.0,
    )

    proposal = model._propose_alternative_reading(seq_hidden, seq_ids, anchor)

    assert proposal["proposal_type"] == "future_window_head"
    assert proposal["proposal_score"] >= cfg.anchor_future_proposal_threshold


def test_alternative_reading_keeps_baseline_on_real_domain_when_future_is_stable():
    cfg = replace(TOY_CONFIG, anchor_domain_mode="real")
    model = ABPTAnchorV1(cfg)
    seq_ids = torch.tensor([12, 13, 12, 13, 12, 13, 12, 13], dtype=torch.long)
    seq_hidden = torch.zeros(seq_ids.size(0), cfg.d_model)
    seq_hidden[:, 0] = 1.0
    anchor = AnchorRecord(
        id=14,
        start_idx=0,
        end_idx=1,
        repr=seq_hidden[0],
        score=0.8,
        state=AnchorState.PROVISIONAL,
        support=0.8,
        contradiction_pressure=0.9,
        viability=0.4,
        ttl=4.0,
    )

    proposal = model._propose_alternative_reading(seq_hidden, seq_ids, anchor)

    assert proposal["proposal_type"] == "start_state_baseline"


def test_real_domain_proposal_trigger_is_lower_than_dead_end():
    cfg = replace(
        TOY_CONFIG,
        anchor_domain_mode="real",
        anchor_dead_end_threshold=0.85,
        anchor_future_proposal_trigger=0.35,
    )
    model = ABPTAnchorV1(cfg)

    assert model._should_request_proposal(pressure=0.40, domain_mode="real")
    assert not model._should_request_proposal(pressure=0.34, domain_mode="real")
    assert not model._should_request_proposal(pressure=0.40, domain_mode="synthetic")


def test_proposal_aux_losses_are_positive_for_non_baseline_real_proposal():
    cfg = replace(TOY_CONFIG, anchor_domain_mode="real", d_model=4)
    model = ABPTAnchorV1(cfg)
    hidden = torch.zeros(1, 6, cfg.d_model)
    hidden[0, 0:2, 0] = 1.0
    hidden[0, 2:4, 1] = 1.0
    hidden[0, 4:6, 1] = 1.0
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[0, 0] = 4.0
        model.lm_head.weight[1, 1] = 4.0
        model.lm_head.weight[1, 0] = -4.0
    targets = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
    anchor = AnchorRecord(
        id=21,
        start_idx=0,
        end_idx=1,
        repr=hidden[0, 0],
        score=0.8,
        state=AnchorState.PROVISIONAL,
        support=0.8,
        contradiction_pressure=0.7,
        viability=0.3,
        ttl=4.0,
        descendant_coherence=0.0,
    )
    losses = model._proposal_aux_losses(
        hidden=hidden,
        anchors=[[anchor]],
        proposal_map={
            21: {
                "repr": hidden[0, 2:4].mean(dim=0),
                "branch_repr": hidden[0, 2:4].mean(dim=0),
                "rollout_states": torch.stack(
                    [hidden[0, 2], hidden[0, 3], hidden[0, 4], hidden[0, 5]],
                    dim=0,
                ),
                "rollout_steps": 4,
                "proposal_type": "future_window_head",
                "proposal_span": (2, 3),
                "proposal_score": 0.8,
                "proposal_score_tensor": torch.tensor(0.8, dtype=hidden.dtype),
            }
        },
        targets=targets,
    )

    assert losses["proposal_score_loss"].item() >= 0.0
    assert losses["proposal_margin_loss"].item() >= 0.0
    assert losses["proposal_alignment_loss"].item() >= 0.0
    assert losses["proposal_counterfactual_loss"].item() >= 0.0
    assert losses["proposal_rollout_loss"].item() >= 0.0
    assert losses["proposal_counterfactual_gain"].item() > 0.0
    assert losses["proposal_rollout_gain"].item() > 0.0
    assert losses["proposal_counterfactual_current_ce"].item() > losses["proposal_counterfactual_proposal_ce"].item()
    assert losses["proposal_counterfactual_count"].item() == 1.0
    assert losses["proposal_rollout_count"].item() == 1.0
    assert losses["proposal_rollout_depth"].item() > 0.0


def test_attach_proposal_rollout_respects_gates():
    cfg = replace(
        TOY_CONFIG,
        anchor_domain_mode="real",
        anchor_proposal_rollout_pressure_trigger=0.45,
        anchor_proposal_rollout_score_trigger=0.90,
    )
    model = ABPTAnchorV1(cfg)
    seq_hidden = torch.randn(8, cfg.d_model)
    anchor = AnchorRecord(
        id=31,
        start_idx=1,
        end_idx=2,
        repr=seq_hidden[2],
        score=0.8,
        state=AnchorState.PROVISIONAL,
        support=0.8,
        contradiction_pressure=0.40,
        viability=0.4,
        ttl=4.0,
    )
    proposal = {
        "repr": seq_hidden[4],
        "proposal_type": "future_window_head",
        "proposal_score": 0.95,
        "proposal_span": (4, 4),
    }

    gated = model._attach_proposal_rollout(seq_hidden=seq_hidden, anchor=anchor, proposal=proposal)
    assert "rollout_states" not in gated

    anchor.contradiction_pressure = 0.60
    proposal["proposal_score"] = 0.50
    gated = model._attach_proposal_rollout(seq_hidden=seq_hidden, anchor=anchor, proposal=proposal)
    assert "rollout_states" not in gated

    proposal["proposal_score"] = 0.95
    applied = model._attach_proposal_rollout(seq_hidden=seq_hidden, anchor=anchor, proposal=proposal)
    assert "rollout_states" in applied
    assert int(applied["rollout_steps"]) > 0


def test_optional_proposal_modules_are_really_toggled_by_config():
    cfg_enabled = replace(
        TOY_CONFIG,
        anchor_domain_mode="real",
        anchor_use_future_proposal_head=True,
        anchor_use_proposal_rollout=True,
    )
    cfg_disabled = replace(
        TOY_CONFIG,
        anchor_domain_mode="real",
        anchor_use_future_proposal_head=False,
        anchor_use_proposal_rollout=False,
    )

    model_enabled = ABPTAnchorV1(cfg_enabled)
    model_disabled = ABPTAnchorV1(cfg_disabled)

    assert model_enabled.future_proposal_head is not None
    assert model_enabled.proposal_rollout is not None
    assert model_disabled.future_proposal_head is None
    assert model_disabled.proposal_rollout is None
    assert model_enabled.param_count() > model_disabled.param_count()
