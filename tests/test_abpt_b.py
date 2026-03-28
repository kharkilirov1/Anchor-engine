import torch
from dataclasses import replace
from src.model.config import TOY_CONFIG
from src.model.abpt_b import ABPTModelB


def test_abpt_b_forward():
    cfg = replace(TOY_CONFIG)
    model = ABPTModelB(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, targets)
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert "loss" in out
    assert "route_stats" in out


def test_abpt_b_backward():
    cfg = replace(TOY_CONFIG)
    model = ABPTModelB(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, targets)
    out["loss"].backward()
    # Router thresholds: non-differentiable (argmax)
    # Plastic/branch: may not get grads if no tokens routed there
    skip_prefixes = ("router.", "plastic.", "branch_router.", "verifier.")
    grads_found = 0
    for n, p in model.named_parameters():
        if p.requires_grad and not any(n.startswith(s) for s in skip_prefixes):
            assert p.grad is not None, f"No gradient for {n}"
            grads_found += 1
    assert grads_found > 0


def test_abpt_b_route_stats():
    cfg = replace(TOY_CONFIG)
    model = ABPTModelB(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x)
    stats = out["route_stats"]
    assert len(stats) == cfg.n_layers
    for s in stats:
        assert "forward" in s
        assert "branch" in s
        assert "backward" in s
        assert "plastic" in s
        assert "mean_ed" in s
        total = s["forward"] + s["branch"] + s["backward"] + s["plastic"]
        assert abs(total - 1.0) < 0.01


def test_abpt_b_no_modules():
    cfg = replace(TOY_CONFIG, use_attn_res=False, use_branches=False,
                  use_verifier=False, use_plastic=False)
    model = ABPTModelB(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, targets)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    out["loss"].backward()


def test_abpt_b_param_count():
    cfg = replace(TOY_CONFIG)
    model = ABPTModelB(cfg)
    print(f"Stage B toy params: {model.param_count_str()}")
    assert model.param_count() < 5_000_000
