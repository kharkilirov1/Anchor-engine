"""Synthetic data for FOG ablation — algorithmic tasks."""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class CopyTask(Dataset):
    """Copy: input [a, b, c, SEP] → output [a, b, c].
    Tests Φ(memory) — pure retrieval."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int = 42):
        super().__init__()
        self.sep_token = vocab_size - 1
        rng = random.Random(seed)
        self.samples = []
        self.sep_positions = []
        content_vocab = vocab_size - 1
        half = seq_len // 2 - 1
        for _ in range(n_samples):
            content = [rng.randint(0, content_vocab - 1) for _ in range(half)]
            ids = content + [self.sep_token] + content
            sep_pos = len(content)
            ids = ids[:seq_len]
            while len(ids) < seq_len:
                ids.append(0)
            self.samples.append(ids)
            self.sep_positions.append(sep_pos)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        # loss_mask: 1 after SEP, 0 before (shifted by -1 for targets)
        mask = torch.zeros_like(y)
        sep = self.sep_positions[idx]
        if sep < len(mask):
            mask[sep:] = 1
        return {"input_ids": x, "targets": y, "loss_mask": mask}


class ReverseTask(Dataset):
    """Reverse: input [a, b, c, SEP] → output [c, b, a].
    Tests Φ(compare) + Φ(memory) — positional matching + retrieval."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int = 42):
        super().__init__()
        self.sep_token = vocab_size - 1
        rng = random.Random(seed)
        self.samples = []
        self.sep_positions = []
        content_vocab = vocab_size - 1
        half = seq_len // 2 - 1
        for _ in range(n_samples):
            content = [rng.randint(0, content_vocab - 1) for _ in range(half)]
            ids = content + [self.sep_token] + list(reversed(content))
            sep_pos = len(content)
            ids = ids[:seq_len]
            while len(ids) < seq_len:
                ids.append(0)
            self.samples.append(ids)
            self.sep_positions.append(sep_pos)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        mask = torch.zeros_like(y)
        sep = self.sep_positions[idx]
        if sep < len(mask):
            mask[sep:] = 1
        return {"input_ids": x, "targets": y, "loss_mask": mask}


class SelectiveRetrieval(Dataset):
    """Context has key-value pairs, query asks for value by key.
    Tests Φ(compare) + Φ(select) + Φ(memory).

    Format: [k1, v1, k2, v2, ..., SEP, query_key, answer_value]
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 4, seed: int = 42):
        super().__init__()
        self.sep_token = vocab_size - 1
        rng = random.Random(seed)
        self.samples = []
        self.sep_positions = []
        content_vocab = vocab_size - 2  # exclude SEP and padding
        for _ in range(n_samples):
            keys = rng.sample(range(content_vocab), min(n_pairs, content_vocab))
            values = [rng.randint(0, content_vocab - 1) for _ in keys]
            query_idx = rng.randint(0, len(keys) - 1)

            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sep_pos = len(ids)
            ids.append(self.sep_token)
            ids.append(keys[query_idx])
            ids.append(values[query_idx])

            ids = ids[:seq_len]
            while len(ids) < seq_len:
                ids.append(0)
            self.samples.append(ids)
            self.sep_positions.append(sep_pos)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        mask = torch.zeros_like(y)
        sep = self.sep_positions[idx]
        if sep < len(mask):
            mask[sep:] = 1
        return {"input_ids": x, "targets": y, "loss_mask": mask}
