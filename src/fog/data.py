"""Synthetic data for FOG ablation — algorithmic tasks.

Easy: CopyTask, ReverseTask, SelectiveRetrieval
Hard: DistractorRetrieval, NoisyRetrieval, MultiQueryRetrieval, ChainedRetrieval
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


def _build_item(ids: list[int], sep_pos: int, seq_len: int) -> dict[str, torch.Tensor]:
    """Shared helper: pad/truncate, build input/target/mask."""
    real_len = len(ids)
    ids = ids[:seq_len]
    ids += [0] * (seq_len - len(ids))
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    m = torch.zeros_like(y)
    # Only mask real tokens after SEP, not padding
    end = min(real_len - 1, len(m))  # -1 because targets are shifted
    if sep_pos < end:
        m[sep_pos:end] = 1
    return {"input_ids": x, "targets": y, "loss_mask": m}


def prebatch_dataset(dataset: Dataset, seq_len: int) -> dict[str, torch.Tensor]:
    """Pre-stack entire dataset into contiguous tensors for fast batching."""
    n = len(dataset)
    all_x = torch.zeros(n, seq_len - 1, dtype=torch.long)
    all_y = torch.zeros(n, seq_len - 1, dtype=torch.long)
    all_m = torch.zeros(n, seq_len - 1, dtype=torch.long)
    for i in range(n):
        item = dataset[i]
        L = item["input_ids"].size(0)
        all_x[i, :L] = item["input_ids"]
        all_y[i, :L] = item["targets"]
        all_m[i, :L] = item["loss_mask"]
    return {"input_ids": all_x, "targets": all_y, "loss_mask": all_m}


class TensorBatchIterator:
    """Fast batch iterator over pre-stacked tensors. No DataLoader overhead."""

    def __init__(self, data: dict[str, torch.Tensor], batch_size: int, shuffle: bool = False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = data["input_ids"].size(0)

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n)
        else:
            perm = torch.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield {k: v[idx] for k, v in self.data.items()}

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size


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


class ChainedRetrieval(Dataset):
    """Two-hop lookup: find value for query key, use that value as key for second lookup.

    [k1,v1, k2,v2, ..., kN,vN, SEP, query_key, final_answer]

    The model must:
    1. Compare query_key against all keys → find matching value (Φ_compare + Φ_memory)
    2. Use that value as a new key → find its value (Φ_compose + Φ_memory)
    3. Output the final value

    This is compositional: uniform models with shared compare/memory
    struggle when capacity is tight, while motif-aware models with
    dedicated compare (narrow) and memory (wide) subspaces can separate
    the two lookups.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int,
                 n_pairs: int = 6, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        attempts = 0
        while len(self.items) < n_samples and attempts < n_samples * 20:
            attempts += 1
            if cv < n_pairs:
                break
            keys = rng.sample(range(cv), n_pairs)
            values = [rng.randint(0, cv - 1) for _ in keys]
            # Find a valid chain: query_key → value_1, value_1 must be a key → value_2
            # value_1 must appear as a key somewhere (different pair)
            chain_found = False
            for qi in range(n_pairs):
                v1 = values[qi]
                for hop2 in range(n_pairs):
                    if hop2 != qi and keys[hop2] == v1:
                        # Chain: query keys[qi] → values[qi]=v1, then v1=keys[hop2] → values[hop2]
                        ids = []
                        for k, v in zip(keys, values):
                            ids.extend([k, v])
                        sp = len(ids)
                        answer = values[hop2]
                        ids += [sep, keys[qi], answer]
                        self.items.append(_build_item(ids, sp, seq_len))
                        chain_found = True
                        break
                if chain_found:
                    break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


# ── Stress tasks for 400-800K models ────────────────────────────


class ConditionalRetrieval(Dataset):
    """Conditional branching: [k1,v1,k2,v2,..., SEP, query_key, threshold] -> value if key>threshold else 0.

    Forces the model to compare query_key against keys AND compare the matched key
    against a threshold — two distinct comparison operations chained with a branch.
    Tests: Phi_compare (double), Phi_memory, conditional logic.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int,
                 n_pairs: int = 6, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(1, cv - 1) for _ in keys]
            qi = rng.randint(0, len(keys) - 1)
            threshold = rng.randint(0, cv - 1)
            answer = values[qi] if keys[qi] > threshold else 0
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sp = len(ids)
            ids += [sep, keys[qi], threshold, answer]
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class SetIntersection(Dataset):
    """Two sets A and B, output sorted intersection.

    [a1,a2,...,aN, SEP1, b1,b2,...,bM, SEP2, sorted intersection tokens...]

    Forces the model to: hold set A in memory, scan B comparing each element,
    collect matches, output them sorted. Multi-output, requires compare+memory+ordering.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int,
                 set_size: int = 8, overlap: int = 3, seed: int = 42):
        super().__init__()
        sep1 = vocab_size - 1
        sep2 = vocab_size - 2
        rng = random.Random(seed)
        cv = vocab_size - 3
        self.items = []
        for _ in range(n_samples):
            pool = rng.sample(range(cv), min(set_size * 2, cv))
            shared = pool[:overlap]
            a_only = pool[overlap:set_size]
            b_only = pool[set_size:set_size + (set_size - overlap)]
            set_a = shared + a_only
            set_b = shared + b_only
            rng.shuffle(set_a)
            rng.shuffle(set_b)
            intersection = sorted(shared)
            ids = set_a + [sep1] + set_b + [sep2] + intersection
            sp = len(set_a) + 1 + len(set_b)  # mask starts after SEP2
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class ComposeArithmetic(Dataset):
    """Retrieve two values by keys, output (v1 + v2) mod M.

    [k1,v1,...,kN,vN, SEP, query_key1, query_key2, answer]

    Forces: two parallel retrievals + arithmetic composition.
    Tests: Phi_memory (dual retrieval), Phi_compose (modular addition).
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int,
                 n_pairs: int = 6, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        modulus = cv  # values stay in valid token range
        self.items = []
        for _ in range(n_samples):
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(0, cv - 1) for _ in keys]
            qi1, qi2 = rng.sample(range(len(keys)), 2)
            answer = (values[qi1] + values[qi2]) % modulus
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sp = len(ids)
            ids += [sep, keys[qi1], keys[qi2], answer]
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class MultiHopChained(Dataset):
    """3-hop chained retrieval with guaranteed chains (no retry loop).

    Constructs chains deterministically: picks 4 keys, wires k0→k1→k2→k3,
    fills remaining pairs randomly. Output = value of final hop.

    Forces: 3 sequential compose operations, each requiring compare+memory.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int,
                 n_pairs: int = 10, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            all_keys = rng.sample(range(cv), min(n_pairs, cv))
            # Wire a guaranteed 3-hop chain: k0→k1, k1→k2, k2→final_answer
            values = [rng.randint(0, cv - 1) for _ in all_keys]
            values[0] = all_keys[1]  # hop1: k0 → k1
            values[1] = all_keys[2]  # hop2: k1 → k2
            values[2] = rng.randint(0, cv - 1)  # hop3: k2 → answer
            answer = values[2]
            # Shuffle pair order so chain isn't positionally obvious
            pairs = list(zip(all_keys, values))
            rng.shuffle(pairs)
            ids = []
            for k, v in pairs:
                ids.extend([k, v])
            sp = len(ids)
            ids += [sep, all_keys[0], answer]
            self.items.append(_build_item(ids, sp, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]
