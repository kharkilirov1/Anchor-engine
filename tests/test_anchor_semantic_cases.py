from dataclasses import replace

import torch

from src.data.anchor_semantic_cases import make_semantic_anchor_cases, semantic_token_legend
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.model.config import TOY_CONFIG


def test_semantic_cases_are_well_formed():
    cases = make_semantic_anchor_cases()
    legend = semantic_token_legend()

    assert len(cases) >= 5
    assert len(legend) >= 8
    for case in cases:
        assert case.input_ids.shape == (24,)
        assert case.target_ids.shape == (24,)
        assert torch.all(case.input_ids >= 0)
        assert torch.all(case.input_ids < TOY_CONFIG.vocab_size)


def test_anchor_v1_runs_on_semantic_cases():
    torch.manual_seed(7)
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.15,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    model = ABPTAnchorV1(cfg)
    model.eval()
    for case in make_semantic_anchor_cases():
        out = model(case.input_ids.unsqueeze(0), case.target_ids.unsqueeze(0))
        diag = out['anchor_diagnostics']
        assert diag['num_active'] >= 0
        assert diag['mean_contradiction_pressure'] >= 0.0
        assert diag['revision_event_count'] == len(out['revision_events'])


def test_semantic_probe_shows_quantifier_separation():
    torch.manual_seed(7)
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.15,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    model = ABPTAnchorV1(cfg)
    model.eval()
    cases = {case.name: case for case in make_semantic_anchor_cases()}

    stable = model(cases['forall_stable'].input_ids.unsqueeze(0), cases['forall_stable'].target_ids.unsqueeze(0))
    conflict = model(cases['forall_exists_conflict'].input_ids.unsqueeze(0), cases['forall_exists_conflict'].target_ids.unsqueeze(0))

    stable_diag = stable['anchor_diagnostics']
    conflict_diag = conflict['anchor_diagnostics']

    assert stable_diag['mean_contradiction_pressure'] < conflict_diag['mean_contradiction_pressure']
    assert stable_diag['mean_viability'] > conflict_diag['mean_viability']
    assert stable_diag['dead_end_count'] < conflict_diag['dead_end_count']



def test_semantic_probe_shows_induction_separation():
    torch.manual_seed(7)
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.15,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    model = ABPTAnchorV1(cfg)
    model.eval()
    cases = {case.name: case for case in make_semantic_anchor_cases()}

    stable = model(cases['induction_stable'].input_ids.unsqueeze(0), cases['induction_stable'].target_ids.unsqueeze(0))
    conflict = model(cases['induction_example_conflict'].input_ids.unsqueeze(0), cases['induction_example_conflict'].target_ids.unsqueeze(0))

    stable_diag = stable['anchor_diagnostics']
    conflict_diag = conflict['anchor_diagnostics']

    assert stable_diag['mean_contradiction_pressure'] < conflict_diag['mean_contradiction_pressure']
    assert stable_diag['mean_viability'] > conflict_diag['mean_viability']



def test_semantic_probe_shows_proof_mode_separation():
    torch.manual_seed(7)
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.15,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    model = ABPTAnchorV1(cfg)
    model.eval()
    cases = {case.name: case for case in make_semantic_anchor_cases()}

    stable = model(cases['contradiction_stable'].input_ids.unsqueeze(0), cases['contradiction_stable'].target_ids.unsqueeze(0))
    conflict = model(cases['contradiction_direct_conflict'].input_ids.unsqueeze(0), cases['contradiction_direct_conflict'].target_ids.unsqueeze(0))

    stable_diag = stable['anchor_diagnostics']
    conflict_diag = conflict['anchor_diagnostics']

    assert stable_diag['mean_contradiction_pressure'] < conflict_diag['mean_contradiction_pressure']
    assert stable_diag['mean_viability'] > conflict_diag['mean_viability']
    assert stable_diag['dead_end_count'] < conflict_diag['dead_end_count']



def test_semantic_probe_shows_epsilon_delta_separation():
    torch.manual_seed(7)
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.15,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    model = ABPTAnchorV1(cfg)
    model.eval()
    cases = {case.name: case for case in make_semantic_anchor_cases()}

    stable = model(cases['epsilon_delta_stable'].input_ids.unsqueeze(0), cases['epsilon_delta_stable'].target_ids.unsqueeze(0))
    conflict = model(cases['epsilon_close_conflict'].input_ids.unsqueeze(0), cases['epsilon_close_conflict'].target_ids.unsqueeze(0))

    stable_diag = stable['anchor_diagnostics']
    conflict_diag = conflict['anchor_diagnostics']

    assert stable_diag['mean_contradiction_pressure'] < conflict_diag['mean_contradiction_pressure']
    assert stable_diag['mean_viability'] > conflict_diag['mean_viability']
