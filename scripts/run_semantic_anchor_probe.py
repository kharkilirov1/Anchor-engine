from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.anchor_semantic_cases import make_semantic_anchor_cases, semantic_token_legend
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.model.config import TOY_CONFIG


def main() -> None:
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

    print('=== Semantic Anchor Probe ===')
    print(f'model_params={model.param_count():,}')
    print('legend=')
    for item in semantic_token_legend():
        print(f'  {item.token_id}: {item.label} ({item.role})')

    with torch.no_grad():
        for case in make_semantic_anchor_cases():
            out = model(case.input_ids.unsqueeze(0), case.target_ids.unsqueeze(0))
            diag = out['anchor_diagnostics']
            proposal_diag = out['proposal_diagnostics']
            actions = [event.action for event in out['revision_events']]
            print()
            print(f'--- {case.name} ---')
            print(f'description={case.description}')
            print(f'expected_failure_mode={case.expected_failure_mode}')
            print(f'num_active={diag["num_active"]}')
            print(f'mean_contradiction_pressure={diag["mean_contradiction_pressure"]:.4f}')
            print(f'mean_viability={diag["mean_viability"]:.4f}')
            print(f'dead_end_count={diag["dead_end_count"]}')
            print(f'proposal_count={proposal_diag["proposal_count"]}')
            print(f'regime_shift_count={proposal_diag["regime_shift_count"]}')
            print(f'anchors_with_proposal_influence={proposal_diag["anchors_with_proposal_influence"]}')
            print(f'mean_proposal_score={proposal_diag["mean_proposal_score"]:.4f}')
            print(f'mean_blend_ratio={proposal_diag["mean_blend_ratio"]:.4f}')
            print(f'proposal_revise_count={proposal_diag["proposal_revise_count"]}')
            print(f'proposal_retire_count={proposal_diag["proposal_retire_count"]}')
            print(f'strong_proposal_retire_count={proposal_diag["strong_proposal_retire_count"]}')
            print(f'mean_strong_retire_gap={proposal_diag["mean_strong_retire_gap"]:.4f}')
            print(f'revision_event_count={diag["revision_event_count"]}')
            print(f'actions={actions[:12]}')


if __name__ == '__main__':
    main()
