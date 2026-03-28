import torch
from dataclasses import replace
from src.model.config import TOY_CONFIG
from src.model.abpt import ABPTModel


def test_abpt_full_forward():
    cfg = replace(TOY_CONFIG)
    model = ABPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, targets)
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert "loss" in out
    assert out["loss"].requires_grad


def test_abpt_backward():
    cfg = replace(TOY_CONFIG)
    model = ABPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, targets)
    out["loss"].backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {n}"


def test_abpt_no_targets():
    cfg = replace(TOY_CONFIG)
    model = ABPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x)
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert "loss" not in out


def test_abpt_param_count():
    cfg = replace(TOY_CONFIG)
    model = ABPTModel(cfg)
    n_params = model.param_count()
    print(f"Toy model params: {n_params:,}")
    assert n_params < 5_000_000
