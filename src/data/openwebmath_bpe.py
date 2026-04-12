from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.the_stack import _normalize_repo_name
from src.data.the_stack_bpe import BPETokenDataset, _train_tokenizer


def _load_openwebmath_text(
    data_dir: str,
    repo_id: str,
    split: str,
    target_bytes: int,
) -> str:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache_name = f"{_normalize_repo_name(repo_id)}_{split}_{target_bytes}.txt"
    cache_path = Path(data_dir) / cache_name
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, streaming=True)
    chunks: list[str] = []
    total = 0
    for sample in ds:
        text = sample.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        block = text + "\n\n"
        chunks.append(block)
        total += len(block.encode("utf-8"))
        if total >= target_bytes:
            break
    if total == 0:
        raise RuntimeError(f"No usable text collected from {repo_id}:{split}.")
    merged = "".join(chunks)
    cache_path.write_text(merged, encoding="utf-8")
    return merged


def load_openwebmath_bpe(
    seq_len: int = 256,
    device: str = "cpu",
    data_dir: str = "data_cache",
    repo_id: str = "open-web-math/open-web-math",
    target_bytes: int = 200_000,
    vocab_size: int = 256,
) -> tuple[BPETokenDataset, BPETokenDataset]:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    prefix = f"{_normalize_repo_name(repo_id)}_{target_bytes}_bpe{vocab_size}"
    tokenizer_path = Path(data_dir) / f"{prefix}_tokenizer.json"
    ids_path = Path(data_dir) / f"{prefix}_ids.pt"
    meta_path = Path(data_dir) / f"{prefix}_meta.json"

    if tokenizer_path.exists() and ids_path.exists() and meta_path.exists():
        token_ids = torch.load(ids_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual_vocab_size = int(meta["vocab_size"])
    else:
        text = _load_openwebmath_text(
            data_dir=data_dir,
            repo_id=repo_id,
            split="train",
            target_bytes=target_bytes,
        )
        tokenizer = _train_tokenizer(text=text, vocab_size=vocab_size)
        token_ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
        actual_vocab_size = tokenizer.get_vocab_size()

        tokenizer.save(str(tokenizer_path))
        torch.save(token_ids, ids_path)
        meta_path.write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "target_bytes": target_bytes,
                    "vocab_size": actual_vocab_size,
                    "token_count": int(token_ids.numel()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    train = BPETokenDataset(
        token_ids=token_ids,
        vocab_size=actual_vocab_size,
        split="train",
        seq_len=seq_len,
        device=device,
    )
    val = BPETokenDataset(
        token_ids=token_ids,
        vocab_size=actual_vocab_size,
        split="val",
        seq_len=seq_len,
        device=device,
    )
    return train, val
