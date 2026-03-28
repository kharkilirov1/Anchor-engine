import json
import os
from pathlib import Path

import torch

from src.data.the_stack import _normalize_repo_name, load_the_stack_text


class BPETokenDataset:
    def __init__(
        self,
        token_ids: torch.Tensor,
        vocab_size: int,
        split: str = "train",
        seq_len: int = 256,
        device: str = "cpu",
        split_data: bool = True,
    ) -> None:
        assert split in ("train", "val")
        self.seq_len = seq_len
        self.device = device
        self._vocab_size = vocab_size

        if split_data:
            n = int(0.9 * len(token_ids))
            self.data = token_ids[:n] if split == "train" else token_ids[n:]
        else:
            self.data = token_ids

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Token corpus too small for seq_len={self.seq_len}. "
                f"Need more than {self.seq_len + 1} tokens, got {len(self.data)}."
            )
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s: s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1: s + self.seq_len + 1] for s in starts])
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.data)


def _cache_prefix(repo_id: str, lang: str, target_bytes: int, vocab_size: int) -> str:
    return f"{_normalize_repo_name(repo_id)}_{lang}_{target_bytes}_bpe{vocab_size}"


def _train_tokenizer(text: str, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text]
    tokenizer.train_from_iterator(lines, trainer=trainer)
    return tokenizer


def load_the_stack_bpe(
    seq_len: int = 256,
    device: str = "cpu",
    data_dir: str = "data_cache",
    repo_id: str = "bigcode/the-stack-smol-xs",
    lang: str = "python",
    target_bytes: int = 12_000_000,
    vocab_size: int = 4096,
) -> tuple[BPETokenDataset, BPETokenDataset]:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    prefix = _cache_prefix(repo_id=repo_id, lang=lang, target_bytes=target_bytes, vocab_size=vocab_size)
    tokenizer_path = Path(data_dir) / f"{prefix}_tokenizer.json"
    ids_path = Path(data_dir) / f"{prefix}_ids.pt"
    meta_path = Path(data_dir) / f"{prefix}_meta.json"

    if tokenizer_path.exists() and ids_path.exists() and meta_path.exists():
        token_ids = torch.load(ids_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual_vocab_size = int(meta["vocab_size"])
    else:
        raw = load_the_stack_text(
            data_dir=data_dir,
            repo_id=repo_id,
            lang=lang,
            target_bytes=target_bytes,
        )
        text = raw.decode("utf-8", errors="ignore")
        tokenizer = _train_tokenizer(text=text, vocab_size=vocab_size)
        encoding = tokenizer.encode(text)
        token_ids = torch.tensor(encoding.ids, dtype=torch.long)
        actual_vocab_size = tokenizer.get_vocab_size()

        tokenizer.save(str(tokenizer_path))
        torch.save(token_ids, ids_path)
        meta_path.write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "lang": lang,
                    "target_bytes": target_bytes,
                    "vocab_size": actual_vocab_size,
                    "token_count": int(token_ids.numel()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    train = BPETokenDataset(token_ids=token_ids, vocab_size=actual_vocab_size, split="train", seq_len=seq_len, device=device)
    val = BPETokenDataset(token_ids=token_ids, vocab_size=actual_vocab_size, split="val", seq_len=seq_len, device=device)
    return train, val
