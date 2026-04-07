"""Tests for FOG baseline and motif-aware models."""
from __future__ import annotations

import torch

from src.fog.config import (
    BASELINE_SMALL, MOTIF_SMALL,
    BASELINE_MICRO, MOTIF_MICRO, UNIFORM_MICRO,
)
from src.fog.model_baseline import BaselineTransformer
from src.fog.model_motif import MotifTransformer
from src.fog.data import (
    CopyTask, ReverseTask, SelectiveRetrieval,
    ChainedRetrieval, prebatch_dataset, TensorBatchIterator,
)


def test_baseline_forward_backward() -> None:
    model = BaselineTransformer(BASELINE_SMALL)
    x = torch.randint(0, BASELINE_SMALL.vocab_size, (2, 32))
    y = torch.randint(0, BASELINE_SMALL.vocab_size, (2, 32))
    out = model(x, y)
    assert out["logits"].shape == (2, 32, BASELINE_SMALL.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()


def test_motif_forward_backward() -> None:
    model = MotifTransformer(MOTIF_SMALL)
    x = torch.randint(0, MOTIF_SMALL.vocab_size, (2, 32))
    y = torch.randint(0, MOTIF_SMALL.vocab_size, (2, 32))
    out = model(x, y)
    assert out["logits"].shape == (2, 32, MOTIF_SMALL.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()


def test_motif_has_separate_subspaces() -> None:
    model = MotifTransformer(MOTIF_SMALL)
    attn = model.blocks[0].attn
    assert attn.q_proj.out_features == MOTIF_SMALL.d_compare
    assert attn.k_proj.out_features == MOTIF_SMALL.d_compare
    assert attn.v_proj.out_features == MOTIF_SMALL.d_memory
    assert MOTIF_SMALL.d_compare < MOTIF_SMALL.d_memory

    ffn = model.blocks[0].ffn
    assert ffn.gate.out_features == MOTIF_SMALL.d_gate
    assert ffn.expand.out_features == MOTIF_SMALL.d_expand
    assert MOTIF_SMALL.d_gate < MOTIF_SMALL.d_expand


def test_copy_task() -> None:
    ds = CopyTask(vocab_size=64, seq_len=32, n_samples=10)
    assert len(ds) == 10
    sample = ds[0]
    assert sample["input_ids"].shape == (31,)
    assert sample["targets"].shape == (31,)


def test_reverse_task() -> None:
    ds = ReverseTask(vocab_size=64, seq_len=32, n_samples=10)
    assert len(ds) == 10


def test_selective_retrieval_task() -> None:
    ds = SelectiveRetrieval(vocab_size=64, seq_len=32, n_samples=10, n_pairs=4)
    assert len(ds) == 10


def test_chained_retrieval_task() -> None:
    ds = ChainedRetrieval(vocab_size=128, seq_len=64, n_samples=50, n_pairs=6)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["input_ids"].shape[0] == 63
    assert sample["targets"].shape[0] == 63
    assert sample["loss_mask"].sum() > 0


def test_prebatch_and_iterator() -> None:
    ds = CopyTask(vocab_size=64, seq_len=32, n_samples=20)
    data = prebatch_dataset(ds, 32)
    assert data["input_ids"].shape == (20, 31)
    loader = TensorBatchIterator(data, batch_size=8, shuffle=True)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0]["input_ids"].shape[0] == 8


def test_micro_configs_forward() -> None:
    for cfg, cls in [
        (BASELINE_MICRO, BaselineTransformer),
        (UNIFORM_MICRO, BaselineTransformer),
        (MOTIF_MICRO, MotifTransformer),
    ]:
        model = cls(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        y = torch.randint(0, cfg.vocab_size, (2, 32))
        out = model(x, y)
        assert out["loss"] is not None
        out["loss"].backward()
