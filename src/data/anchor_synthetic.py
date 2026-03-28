from __future__ import annotations

import random

import torch

from src.data.anchor_semantic_cases import make_semantic_anchor_cases


class AnchorSyntheticDataset:
    def __init__(
        self,
        split: str = "train",
        seq_len: int = 24,
        device: str = "cpu",
        train_repeats: int = 64,
        val_repeats: int = 16,
    ):
        assert split in ("train", "val")
        self.device = device
        self.seq_len = seq_len
        base_cases = make_semantic_anchor_cases(seq_len=seq_len)
        repeats = train_repeats if split == "train" else val_repeats
        self.samples = [
            (case.input_ids.clone(), case.target_ids.clone(), case.name)
            for _ in range(repeats)
            for case in base_cases
        ]
        self._rng = random.Random(7 if split == "train" else 11)

    @property
    def vocab_size(self) -> int:
        max_token = max(int(sample[0].max().item()) for sample in self.samples)
        return max_token + 1

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        chosen = [self.samples[self._rng.randrange(len(self.samples))] for _ in range(batch_size)]
        x = torch.stack([sample[0] for sample in chosen]).to(self.device)
        y = torch.stack([sample[1] for sample in chosen]).to(self.device)
        return x, y

    def __len__(self) -> int:
        return len(self.samples)


def load_anchor_synthetic(
    seq_len: int = 24,
    device: str = "cpu",
) -> tuple[AnchorSyntheticDataset, AnchorSyntheticDataset]:
    train = AnchorSyntheticDataset(split="train", seq_len=seq_len, device=device)
    val = AnchorSyntheticDataset(split="val", seq_len=seq_len, device=device)
    return train, val
