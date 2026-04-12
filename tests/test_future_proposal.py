from dataclasses import replace

import torch

from src.model.anchor_types import AnchorRecord, AnchorState
from src.model.config import TOY_CONFIG
from src.model.future_proposal import FutureProposalHead


def _make_anchor(cfg, repr_vec: torch.Tensor) -> AnchorRecord:
    return AnchorRecord(
        id=1,
        start_idx=0,
        end_idx=1,
        repr=repr_vec,
        score=0.9,
        state=AnchorState.PROVISIONAL,
        support=0.9,
        contradiction_pressure=0.95,
        viability=0.2,
        ttl=4.0,
        descendant_coherence=0.0,
    )


def test_future_proposal_head_selects_conflict_coherent_future_window():
    cfg = replace(TOY_CONFIG, d_model=4)
    head = FutureProposalHead(cfg)
    seq_hidden = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    seq_ids = torch.tensor([10, 10, 21, 22, 23, 23], dtype=torch.long)
    anchor = _make_anchor(cfg, repr_vec=seq_hidden[0])

    proposal = head.propose(seq_hidden=seq_hidden, seq_ids=seq_ids, anchor=anchor)

    assert proposal is not None
    assert proposal["proposal_type"] == "future_window_head"
    assert proposal["proposal_score"] >= cfg.anchor_future_proposal_threshold
    assert proposal["proposal_span"][0] >= 2
    assert proposal["repr"].shape == (cfg.d_model,)


def test_future_proposal_head_rejects_stable_future_window():
    cfg = replace(TOY_CONFIG, d_model=4)
    head = FutureProposalHead(cfg)
    seq_hidden = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    seq_ids = torch.tensor([10, 10, 10, 10, 10], dtype=torch.long)
    anchor = _make_anchor(cfg, repr_vec=seq_hidden[0])

    proposal = head.propose(seq_hidden=seq_hidden, seq_ids=seq_ids, anchor=anchor)

    assert proposal is None
