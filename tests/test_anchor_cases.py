from dataclasses import replace

import torch

from src.data.anchor_cases import make_anchor_probe_cases
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.model.config import TOY_CONFIG


def test_anchor_cases_shape_and_range():
    cases = make_anchor_probe_cases(seq_len=24, vocab_size=TOY_CONFIG.vocab_size)

    assert len(cases) >= 4
    for case in cases:
        assert case.input_ids.shape == (24,)
        assert case.target_ids.shape == (24,)
        assert torch.all(case.input_ids >= 0)
        assert torch.all(case.input_ids < TOY_CONFIG.vocab_size)


def test_anchor_v1_runs_on_probe_cases():
    torch.manual_seed(7)
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.15,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    model = ABPTAnchorV1(cfg)
    cases = make_anchor_probe_cases(seq_len=24, vocab_size=cfg.vocab_size)

    for case in cases:
        out = model(case.input_ids.unsqueeze(0), case.target_ids.unsqueeze(0))
        diagnostics = out['anchor_diagnostics']

        assert 'anchor_candidates' in out
        assert 'revision_events' in out
        assert diagnostics['num_active'] >= 0
        assert diagnostics['mean_contradiction_pressure'] >= 0.0
        assert diagnostics['revision_event_count'] == len(out['revision_events'])
        assert len(out['anchor_candidates']) == 1


def test_anchor_probe_shows_basic_separation():
    torch.manual_seed(7)
    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.15,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.20,
        anchor_dead_end_threshold=0.35,
    )
    model = ABPTAnchorV1(cfg)
    cases = {case.name: case for case in make_anchor_probe_cases(seq_len=24, vocab_size=cfg.vocab_size)}

    stable = model(cases['stable_regime'].input_ids.unsqueeze(0), cases['stable_regime'].target_ids.unsqueeze(0))
    complex_case = model(cases['complexity_conflict'].input_ids.unsqueeze(0), cases['complexity_conflict'].target_ids.unsqueeze(0))

    stable_diag = stable['anchor_diagnostics']
    complex_diag = complex_case['anchor_diagnostics']

    assert stable_diag['mean_contradiction_pressure'] < complex_diag['mean_contradiction_pressure']
    assert stable_diag['mean_viability'] > complex_diag['mean_viability']
    assert stable_diag['mean_descendant_coherence'] > complex_diag['mean_descendant_coherence']
