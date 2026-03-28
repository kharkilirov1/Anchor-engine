"""Tiny Shakespeare dataset loader with character-level tokenization.

Standard benchmark for small LMs (Karpathy / nanoGPT).
~1MB of text, character-level vocab (~65 unique chars).
"""
import torch
import os


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


class ShakespeareDataset:
    """Tiny Shakespeare — character-level language modeling dataset.

    Args:
        path: path to tiny_shakespeare.txt
        split: 'train' (90%) or 'val' (10%)
        seq_len: context window length
        device: torch device
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        seq_len: int = 256,
        device: str = "cpu",
    ):
        assert split in ("train", "val")
        self.seq_len = seq_len
        self.device = device

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = CharTokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # 90/10 train/val split
        n = int(0.9 * len(data))
        self.data = data[:n] if split == "train" else data[n:]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random batch of (input, target) sequences.

        Returns:
            x: [B, T] input token ids
            y: [B, T] target token ids (shifted by 1)
        """
        max_start = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s: s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1: s + self.seq_len + 1] for s in starts])
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.data)


def load_shakespeare(
    seq_len: int = 256,
    device: str = "cpu",
    data_dir: str = "data_cache",
) -> tuple["ShakespeareDataset", "ShakespeareDataset"]:
    """Load train and val splits of Tiny Shakespeare.

    Returns:
        (train_dataset, val_dataset)
    """
    path = os.path.join(data_dir, "tiny_shakespeare.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"tiny_shakespeare.txt not found at {path}. "
            "Place the file in data_cache/ directory."
        )
    train = ShakespeareDataset(path, split="train", seq_len=seq_len, device=device)
    val = ShakespeareDataset(path, split="val", seq_len=seq_len, device=device)
    return train, val
