from dataclasses import replace

import torch

from src.model.anchor_detector import AnchorDetector
from src.model.config import TOY_CONFIG


def test_anchor_detector_shapes():
    cfg = replace(TOY_CONFIG, anchor_threshold=0.5)
    detector = AnchorDetector(cfg)
    hidden = torch.randn(2, 16, cfg.d_model)
    history = torch.roll(hidden, shifts=1, dims=1)
    history[:, 0] = hidden[:, 0]

    out = detector(hidden, history)

    assert out["scores"].shape == (2, 16)
    assert out["semantic_weights"].shape == (2, 16)
    assert out["span_bounds"].shape == (2, 16, 2)
    assert len(out["candidates"]) == 2


def test_anchor_detector_emits_candidates():
    cfg = replace(TOY_CONFIG, anchor_threshold=0.1)
    detector = AnchorDetector(cfg)
    hidden = torch.randn(1, 8, cfg.d_model)
    history = torch.roll(hidden, shifts=1, dims=1)
    history[:, 0] = hidden[:, 0]

    out = detector(hidden, history)

    assert len(out["candidates"][0]) > 0
