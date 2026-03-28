import json
from pathlib import Path

import torch

from src.data.the_stack import _normalize_repo_name
from src.data.the_stack_bpe import BPETokenDataset, _train_tokenizer


def _load_tinystories_text(
    data_dir: str,
    repo_id: str,
    filename: str,
    target_bytes: int,
) -> str:
    from huggingface_hub import hf_hub_download

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache_name = f"{_normalize_repo_name(repo_id)}_{filename.replace('.', '_')}_{target_bytes}.txt"
    cache_path = Path(data_dir) / cache_name
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    src_path = hf_hub_download(repo_id, filename, repo_type="dataset")
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read(target_bytes)
    cache_path.write_text(text, encoding="utf-8")
    return text


def load_tinystories_bpe(
    seq_len: int = 256,
    device: str = "cpu",
    data_dir: str = "data_cache",
    repo_id: str = "roneneldan/TinyStories",
    train_filename: str = "TinyStories-train.txt",
    val_filename: str = "TinyStories-valid.txt",
    target_bytes: int = 16_000_000,
    vocab_size: int = 4096,
) -> tuple[BPETokenDataset, BPETokenDataset]:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    prefix = f"{_normalize_repo_name(repo_id)}_tinystories_{target_bytes}_bpe{vocab_size}"
    tokenizer_path = Path(data_dir) / f"{prefix}_tokenizer.json"
    train_ids_path = Path(data_dir) / f"{prefix}_train_ids.pt"
    val_ids_path = Path(data_dir) / f"{prefix}_val_ids.pt"
    meta_path = Path(data_dir) / f"{prefix}_meta.json"

    if tokenizer_path.exists() and train_ids_path.exists() and val_ids_path.exists() and meta_path.exists():
        train_ids = torch.load(train_ids_path, map_location="cpu")
        val_ids = torch.load(val_ids_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual_vocab_size = int(meta["vocab_size"])
    else:
        train_text = _load_tinystories_text(
            data_dir=data_dir,
            repo_id=repo_id,
            filename=train_filename,
            target_bytes=target_bytes,
        )
        val_text = _load_tinystories_text(
            data_dir=data_dir,
            repo_id=repo_id,
            filename=val_filename,
            target_bytes=max(1_000_000, target_bytes // 8),
        )
        tokenizer = _train_tokenizer(text=train_text, vocab_size=vocab_size)
        train_ids = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
        val_ids = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)
        actual_vocab_size = tokenizer.get_vocab_size()

        tokenizer.save(str(tokenizer_path))
        torch.save(train_ids, train_ids_path)
        torch.save(val_ids, val_ids_path)
        meta_path.write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "target_bytes": target_bytes,
                    "vocab_size": actual_vocab_size,
                    "train_token_count": int(train_ids.numel()),
                    "val_token_count": int(val_ids.numel()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    train = BPETokenDataset(
        token_ids=train_ids,
        vocab_size=actual_vocab_size,
        split="train",
        seq_len=seq_len,
        device=device,
        split_data=False,
    )
    val = BPETokenDataset(
        token_ids=val_ids,
        vocab_size=actual_vocab_size,
        split="val",
        seq_len=seq_len,
        device=device,
        split_data=False,
    )
    return train, val
