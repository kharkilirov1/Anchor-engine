from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_probe_cases import make_qwen_probe_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anchor diagnostics on top of Qwen hidden states.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    cfg = replace(
        TOY_CONFIG,
        anchor_threshold=0.20,
        anchor_revision_threshold=0.45,
        anchor_contradiction_threshold=0.25,
        anchor_dead_end_threshold=0.40,
    )
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model,
        cfg=cfg,
        device=args.device,
        torch_dtype=torch.float16 if "cuda" in args.device else None,
    )
    overlay.eval()

    print("=== Qwen Anchor Probe ===")
    print(f"model={args.model}")
    print(f"device={args.device}")
    print()

    for case in make_qwen_probe_cases():
        out, batch = overlay.analyze_texts([case.prompt], max_length=args.max_length)
        diag = out["anchor_diagnostics"]
        proposal_diag = out["proposal_diagnostics"]
        print(f"--- {case.name} ---")
        print(f"description={case.description}")
        print(f"expected_mode={case.expected_mode}")
        print(f"tokens={int(batch['input_ids'].numel())}")
        print(f"num_active={diag['num_active']}")
        print(f"mean_contradiction_pressure={diag['mean_contradiction_pressure']:.4f}")
        print(f"mean_viability={diag['mean_viability']:.4f}")
        print(f"dead_end_count={diag['dead_end_count']}")
        print(f"proposal_count={proposal_diag['proposal_count']}")
        print()


if __name__ == "__main__":
    main()
