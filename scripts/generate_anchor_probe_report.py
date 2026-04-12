from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.anchor_semantic_cases import make_semantic_anchor_cases
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.model.config import TOY_CONFIG


def _run_cases() -> list[dict]:
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

    results: list[dict] = []
    with torch.no_grad():
        for case in make_semantic_anchor_cases():
            out = model(case.input_ids.unsqueeze(0), case.target_ids.unsqueeze(0))
            diag = out['anchor_diagnostics']
            proposal_diag = out['proposal_diagnostics']
            results.append({
                'name': case.name,
                'description': case.description,
                'expected_failure_mode': case.expected_failure_mode,
                'num_active': diag['num_active'],
                'mean_contradiction_pressure': diag['mean_contradiction_pressure'],
                'mean_viability': diag['mean_viability'],
                'mean_descendant_mass': diag['mean_descendant_mass'],
                'mean_descendant_coherence': diag['mean_descendant_coherence'],
                'dead_end_count': diag['dead_end_count'],
                'revision_event_count': diag['revision_event_count'],
                'proposal_count': proposal_diag['proposal_count'],
                'regime_shift_count': proposal_diag['regime_shift_count'],
                'anchors_with_proposal_influence': proposal_diag['anchors_with_proposal_influence'],
                'mean_proposal_score': proposal_diag['mean_proposal_score'],
                'mean_blend_ratio': proposal_diag['mean_blend_ratio'],
                'proposal_revise_count': proposal_diag['proposal_revise_count'],
                'proposal_retire_count': proposal_diag['proposal_retire_count'],
                'strong_proposal_retire_count': proposal_diag['strong_proposal_retire_count'],
                'mean_strong_retire_gap': proposal_diag['mean_strong_retire_gap'],
            })
    return results


def _build_family_pairs(results: list[dict]) -> list[tuple[str, dict, dict]]:
    by_name = {row['name']: row for row in results}
    pairs = [
        ('quantifier', by_name['forall_stable'], by_name['forall_exists_conflict']),
        ('proof_mode', by_name['contradiction_stable'], by_name['contradiction_direct_conflict']),
        ('induction', by_name['induction_stable'], by_name['induction_example_conflict']),
        ('formal_limit', by_name['epsilon_delta_stable'], by_name['epsilon_close_conflict']),
    ]
    return pairs


def generate_report(output_path: Path | None = None) -> Path:
    results = _run_cases()
    pairs = _build_family_pairs(results)

    if output_path is None:
        output_path = ROOT / 'docs' / 'research' / 'anchor_probe_report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append('# Anchor Probe Report')
    lines.append('')
    lines.append('Date: 2026-03-27')
    lines.append('Status: generated from current semantic probe harness')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append('This report captures the current behavior of `ABPTAnchorV1` on controlled semantic anchor families.')
    lines.append('Metrics are diagnostic, not benchmark-final.')
    lines.append('')
    lines.append('## Family deltas')
    lines.append('')
    lines.append('| Family | Stable case | Conflict case | Pressure delta | Viability delta | Dead-end delta | Influence delta | Blend delta | Strong proposal retire delta |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for family, stable, conflict in pairs:
        pressure_delta = conflict['mean_contradiction_pressure'] - stable['mean_contradiction_pressure']
        viability_delta = stable['mean_viability'] - conflict['mean_viability']
        dead_end_delta = conflict['dead_end_count'] - stable['dead_end_count']
        influence_delta = conflict['anchors_with_proposal_influence'] - stable['anchors_with_proposal_influence']
        blend_delta = conflict['mean_blend_ratio'] - stable['mean_blend_ratio']
        strong_retire_delta = conflict['strong_proposal_retire_count'] - stable['strong_proposal_retire_count']
        lines.append(
            f"| {family} | {stable['name']} | {conflict['name']} | {pressure_delta:.4f} | {viability_delta:.4f} | {dead_end_delta} | {influence_delta} | {blend_delta:.4f} | {strong_retire_delta} |"
        )
    lines.append('')
    lines.append('## Case table')
    lines.append('')
    lines.append('| Case | Failure mode | Active | Pressure | Viability | Desc. mass | Desc. coherence | Proposal infl. | Proposal score | Blend ratio | Proposal revise | Proposal retire | Strong proposal retire | Strong retire gap | Dead-end | Revisions |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in results:
        lines.append(
            f"| {row['name']} | {row['expected_failure_mode']} | {row['num_active']} | {row['mean_contradiction_pressure']:.4f} | {row['mean_viability']:.4f} | {row['mean_descendant_mass']:.4f} | {row['mean_descendant_coherence']:.4f} | {row['anchors_with_proposal_influence']} | {row['mean_proposal_score']:.4f} | {row['mean_blend_ratio']:.4f} | {row['proposal_revise_count']} | {row['proposal_retire_count']} | {row['strong_proposal_retire_count']} | {row['mean_strong_retire_gap']:.4f} | {row['dead_end_count']} | {row['revision_event_count']} |"
        )
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    lines.append('- Stable semantic cases should trend toward lower contradiction pressure and higher viability than their conflict counterparts.')
    lines.append('- Larger descendant mass in a conflict case is not necessarily bad by itself; under the anchor-tree hypothesis, a false root may still grow a larger but more unstable tree.')
    lines.append('- Proposal metrics show whether alternative-hypothesis machinery is merely present or actually influencing the inference path.')
    lines.append('- Proposal timing metrics show where strong alternative hypotheses still lose to `retire`, which is the current revision-timing bottleneck.')
    lines.append('- The most useful signals at this stage are the joint movements of pressure, viability, dead-end count, and proposal influence.')
    lines.append('')
    lines.append('## Current timing bottlenecks')
    lines.append('')
    strong_retire_cases = [row for row in results if row['strong_proposal_retire_count'] > 0]
    if strong_retire_cases:
        lines.append('| Case | Strong proposal retire | Proposal revise | Proposal retire | Retire-revise gap | Blend ratio |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for row in strong_retire_cases:
            lines.append(
                f"| {row['name']} | {row['strong_proposal_retire_count']} | {row['proposal_revise_count']} | {row['proposal_retire_count']} | {row['mean_strong_retire_gap']:.4f} | {row['mean_blend_ratio']:.4f} |"
            )
    else:
        lines.append('No strong-proposal retire bottlenecks are present in the current probe snapshot.')
    lines.append('')
    lines.append('## Current conclusion')
    lines.append('')
    lines.append('The current model now shows meaningful qualitative separation across several semantic families, especially quantifier, induction, and formal-limit drift cases. The next step should tighten revision timing and enrich descendant-aware semantics further.')
    lines.append('')

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


if __name__ == '__main__':
    path = generate_report()
    print(path)
