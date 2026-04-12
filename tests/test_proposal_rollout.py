from dataclasses import replace

import torch

from src.model.config import TOY_CONFIG
from src.model.proposal_rollout import ProposalRolloutBranch


def test_proposal_rollout_branch_shapes():
    cfg = replace(TOY_CONFIG, d_model=16, anchor_proposal_rollout_steps=3, anchor_proposal_rollout_hidden=32)
    branch = ProposalRolloutBranch(cfg)
    anchor_repr = torch.randn(cfg.d_model)
    proposal_repr = torch.randn(cfg.d_model)
    context_repr = torch.randn(cfg.d_model)

    out = branch(anchor_repr=anchor_repr, proposal_repr=proposal_repr, context_repr=context_repr)

    assert out["states"].shape == (3, cfg.d_model)
    assert out["summary"].shape == (cfg.d_model,)
    assert torch.isfinite(out["states"]).all()
    assert torch.isfinite(out["summary"]).all()
