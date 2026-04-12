from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.anchor_cases import make_anchor_probe_cases
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

    cases = make_anchor_probe_cases(seq_len=24, vocab_size=cfg.vocab_size)
    print('=== Anchor Probe ===')
    print(f'model_params={model.param_count():,}')
    print(f'num_cases={len(cases)}')

    with torch.no_grad():
        for case in cases:
            x = case.input_ids.unsqueeze(0)
            y = case.target_ids.unsqueeze(0)
            out = model(x, y)
            diagnostics = out['anchor_diagnostics']
            actions = [event.action for event in out['revision_events']]
            reasons = [event.reason for event in out['revision_events']]
            print()
            print(f'--- {case.name} ---')
            print(f'description={case.description}')
            print(f'expected_failure_mode={case.expected_failure_mode}')
            print(f'loss={out["loss"].item():.4f}')
            print(f'num_active={diagnostics["num_active"]}')
            print(f'mean_anchor_score={diagnostics["mean_anchor_score"]:.4f}')
            print(f'mean_contradiction_pressure={diagnostics["mean_contradiction_pressure"]:.4f}')
            print(f'mean_viability={diagnostics["mean_viability"]:.4f}')
            print(f'dead_end_count={diagnostics["dead_end_count"]}')
            print(f'revision_event_count={diagnostics["revision_event_count"]}')
            print(f'actions={actions[:12]}')
            print(f'reasons={reasons[:12]}')


if __name__ == '__main__':
    main()
