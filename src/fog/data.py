"""Synthetic data for FOG ablation — algorithmic tasks.

Easy: CopyTask, ReverseTask, SelectiveRetrieval
Hard: DistractorRetrieval, NoisyRetrieval, MultiQueryRetrieval
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


def _build_item(ids: list[int], sep_pos: int, seq_len: int) -> dict[str, torch.Tensor]:
    """Shared helper: pad/truncate, build input/target/mask."""
    ids = ids[:seq_len]
    ids += [0] * (seq_len - len(ids))
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    m = torch.zeros_like(y)
    if sep_pos < len(m):
        m[sep_pos:] = 1
    return {"input_ids": x, "targets": y, "loss_mask": m}


# ── Easy tasks ──────────────────────────────────────────────────


class CopyTask(Dataset):
    """Copy: [a, b, c, SEP] -> [a, b, c]. Tests memory."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 1
        half = seq_len // 2 - 1
        self.items = []
        for _ in range(n_samples):
            c = [rng.randint(0, cv - 1) for _ in range(half)]
            ids = c + [sep] + c
            self.items.append(_build_item(ids, len(c), seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class ReverseTask(Dataset):
    """Reverse: [a, b, c, SEP] -> [c, b, a]. Tests compare + memory."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 1
        half = seq_len // 2 - 1
        self.items = []
        for _ in range(n_samples):
            c = [rng.randint(0, cv - 1) for _ in range(half)]
            ids = c + [sep] + list(reversed(c))
            self.items.append(_build_item(ids, len(c), seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class SelectiveRetrieval(Dataset):
    """[k1, v1, k2, v2, SEP, query_key] -> answer_value. Tests compare + select + memory."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 4, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(0, cv - 1) for _ in keys]
            qi = rng.randint(0, len(keys) - 1)
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sp = len(ids)
            ids += [sep, keys[qi], values[qi]]
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


# ── Hard tasks ──────────────────────────────────────────────────


class DistractorRetrieval(Dataset):
    """Keys differ by +/-1 from query. Forces precise compare."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 4, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            qk = rng.randint(n_pairs, cv - n_pairs - 1)
            offsets = [i for i in range(-n_pairs, n_pairs + 1) if i != 0]
            dk = [qk + o for o in offsets if 0 <= qk + o < cv]
            keys = [qk] + dk[:n_pairs - 1]
            rng.shuffle(keys)
            values = [rng.randint(0, cv - 1) for _ in keys]
            qi = keys.index(qk)
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sp = len(ids)
            ids += [sep, qk, values[qi]]
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class NoisyRetrieval(Dataset):
    """Noise tokens between KV pairs. Forces select to filter, memory to retrieve."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int,
                 n_pairs: int = 3, noise_len: int = 2, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(0, cv - 1) for _ in keys]
            qi = rng.randint(0, len(keys) - 1)
            ids = []
            for i, (k, v) in enumerate(zip(keys, values)):
                ids.extend([k, v])
                if i < len(keys) - 1:
                    ids.extend([rng.randint(0, cv - 1) for _ in range(noise_len)])
            sp = len(ids)
            ids += [sep, keys[qi], values[qi]]
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class MultiQueryRetrieval(Dataset):
    """2 sequential queries — retrieve 2 values. Tests compose."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int,
                 n_pairs: int = 4, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(0, cv - 1) for _ in keys]
            qis = rng.sample(range(len(keys)), min(2, len(keys)))
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sp = len(ids)
            ids.append(sep)
            for qi in qis:
                ids += [keys[qi], values[qi]]
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]
