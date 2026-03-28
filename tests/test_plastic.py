import torch
from dataclasses import replace
from src.model.config import TOY_CONFIG
from src.model.plastic import PlasticLayer


def test_plastic_output_shape():
    cfg = replace(TOY_CONFIG)
    plastic = PlasticLayer(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = plastic(x)
    assert out.shape == (2, 16, cfg.d_model)


def test_plastic_adapts_at_inference():
    cfg = replace(TOY_CONFIG)
    plastic = PlasticLayer(cfg)
    plastic.eval()
    x = torch.randn(1, 8, cfg.d_model)
    out1 = plastic(x).clone()
    plastic.adapt_step(x, lr=cfg.plastic_lr)
    out2 = plastic(x)
    assert not torch.allclose(out1, out2, atol=1e-7)


def test_plastic_decay():
    cfg = replace(TOY_CONFIG)
    plastic = PlasticLayer(cfg)
    with torch.no_grad():
        for p in plastic.adapter.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = {n: p.clone() for n, p in plastic.adapter.named_parameters()}
    plastic.apply_decay(cfg.plastic_decay)
    for n, p in plastic.adapter.named_parameters():
        diff_before = (before[n] - plastic.initial_state[n]).abs().mean()
        diff_after = (p - plastic.initial_state[n]).abs().mean()
        assert diff_after < diff_before


def test_plastic_reset():
    cfg = replace(TOY_CONFIG)
    plastic = PlasticLayer(cfg)
    original = {n: p.clone() for n, p in plastic.adapter.named_parameters()}
    with torch.no_grad():
        for p in plastic.adapter.parameters():
            p.add_(torch.randn_like(p))
    plastic.reset()
    for n, p in plastic.adapter.named_parameters():
        assert torch.allclose(p, original[n])
