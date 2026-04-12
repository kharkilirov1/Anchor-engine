import torch
import pytest
from dataclasses import replace
from src.model.config import PRESETS
from src.model.abpt import ABPTModel


@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_preset_forward_backward(preset_name):
    cfg = replace(PRESETS[preset_name])
    model = ABPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, targets)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    out["loss"].backward()
    print(f"{preset_name}: {model.param_count_str()} params, loss={out['loss'].item():.4f}")
