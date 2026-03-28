import torch
from dataclasses import replace
from src.model.config import TOY_CONFIG
from src.model.branches import BranchRouter


def test_branch_router_output_shape():
    cfg = replace(TOY_CONFIG)
    router = BranchRouter(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = router(x)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert len(out["branch_logits"]) == cfg.n_branches


def test_branch_logits_shapes():
    cfg = replace(TOY_CONFIG)
    router = BranchRouter(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = router(x)
    for bl in out["branch_logits"]:
        assert bl.shape == (2, 16, cfg.vocab_size)


def test_diversity_loss_nonneg():
    cfg = replace(TOY_CONFIG)
    router = BranchRouter(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = router(x)
    assert out["diversity_loss"].item() >= 0.0


def test_branch_temperatures_differ():
    cfg = replace(TOY_CONFIG, n_branches=3)
    router = BranchRouter(cfg)
    temps = [b.temperature for b in router.branches]
    assert len(set(temps)) == 3
