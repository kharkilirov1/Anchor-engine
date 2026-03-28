import os
import json
from pathlib import Path

import torch


class ByteCorpusDataset:
    def __init__(
        self,
        data: bytes,
        split: str = "train",
        seq_len: int = 256,
        device: str = "cpu",
    ) -> None:
        assert split in ("train", "val")
        self.seq_len = seq_len
        self.device = device

        tensor = torch.tensor(list(data), dtype=torch.long)
        n = int(0.9 * len(tensor))
        self.data = tensor[:n] if split == "train" else tensor[n:]

    @property
    def vocab_size(self) -> int:
        return 256

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Corpus too small for seq_len={self.seq_len}. "
                f"Need more than {self.seq_len + 1} bytes, got {len(self.data)}."
            )
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s: s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1: s + self.seq_len + 1] for s in starts])
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.data)


def _normalize_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("-", "_")


def _stream_the_stack_text(
    repo_id: str,
    lang: str,
    target_bytes: int,
) -> bytes:
    try:
        from huggingface_hub import hf_hub_download

        json_path = hf_hub_download(repo_id, f"data/{lang}/data.json", repo_type="dataset")
        chunks: list[bytes] = []
        total = 0
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                text = sample.get("content") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                encoded = text.encode("utf-8", errors="ignore") + b"\n\n"
                chunks.append(encoded)
                total += len(encoded)
                if total >= target_bytes:
                    break
        if total > 0:
            return b"".join(chunks)
    except Exception:
        pass

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets package is required for The Stack loading. "
            "Install with `pip install datasets`."
        ) from exc

    try:
        ds = load_dataset(repo_id, lang, split="train", streaming=True)
    except Exception:
        data_dir = f"data/{lang}"
        try:
            ds = load_dataset(repo_id, data_dir=data_dir, split="train", streaming=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load repo={repo_id} lang={lang}. "
                "If this is a gated dataset, accept the Hugging Face terms first or "
                "switch to a public Stack-family subset such as `bigcode/the-stack-smol-xs`."
            ) from exc

    chunks: list[bytes] = []
    total = 0
    for sample in ds:
        text = sample.get("content") or sample.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = text.encode("utf-8", errors="ignore") + b"\n\n"
        chunks.append(encoded)
        total += len(encoded)
        if total >= target_bytes:
            break

    if total == 0:
        raise RuntimeError(f"No text content collected from repo={repo_id} lang={lang}.")
    return b"".join(chunks)


def load_the_stack_text(
    data_dir: str = "data_cache",
    repo_id: str = "bigcode/the-stack-smol-xs",
    lang: str = "python",
    target_bytes: int = 8_000_000,
) -> bytes:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache_name = f"{_normalize_repo_name(repo_id)}_{lang}_{target_bytes}.bin"
    cache_path = os.path.join(data_dir, cache_name)

    if os.path.exists(cache_path):
        return Path(cache_path).read_bytes()

    data = _stream_the_stack_text(repo_id=repo_id, lang=lang, target_bytes=target_bytes)
    Path(cache_path).write_bytes(data)
    return data


def load_the_stack(
    seq_len: int = 256,
    device: str = "cpu",
    data_dir: str = "data_cache",
    repo_id: str = "bigcode/the-stack-smol-xs",
    lang: str = "python",
    target_bytes: int = 8_000_000,
) -> tuple[ByteCorpusDataset, ByteCorpusDataset]:
    data = load_the_stack_text(
        data_dir=data_dir,
        repo_id=repo_id,
        lang=lang,
        target_bytes=target_bytes,
    )
    train = ByteCorpusDataset(data=data, split="train", seq_len=seq_len, device=device)
    val = ByteCorpusDataset(data=data, split="val", seq_len=seq_len, device=device)
    return train, val
