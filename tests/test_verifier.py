import torch
from dataclasses import replace
from src.model.config import TOY_CONFIG
from src.model.verifier import Verifier


def test_verifier_output_shape():
    cfg = replace(TOY_CONFIG)
    verifier = Verifier(cfg)
    branch_logits = [torch.randn(2, 16, cfg.vocab_size) for _ in range(2)]
    out = verifier(branch_logits)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["confidence"].shape == (2, 16)
    assert out["branch_weights"].shape == (2, 16, 2)


def test_verifier_weights_sum_to_one():
    cfg = replace(TOY_CONFIG)
    verifier = Verifier(cfg)
    branch_logits = [torch.randn(2, 16, cfg.vocab_size) for _ in range(2)]
    out = verifier(branch_logits)
    sums = out["branch_weights"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_verifier_confident_branch_dominates():
    cfg = replace(TOY_CONFIG)
    verifier = Verifier(cfg)
    b0 = torch.zeros(1, 4, cfg.vocab_size)
    b0[:, :, 0] = 10.0  # peaked = low entropy
    b1 = torch.zeros(1, 4, cfg.vocab_size)  # flat = high entropy
    out = verifier([b0, b1])
    assert out["branch_weights"][0, 0, 0] > out["branch_weights"][0, 0, 1]
