from __future__ import annotations

from dataclasses import dataclass
import random

import torch

DOMAIN_NAMES: tuple[str, ...] = ("text", "retrieval", "code")
DOMAIN_TOKEN_MAP: dict[str, int] = {"text": 1, "retrieval": 2, "code": 3}


@dataclass
class QwenMotifSyntheticDataset:
    seq_len: int = 32
    vocab_size: int = 64
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.device_obj = torch.device(self.device)

    def _make_text_sequence(self) -> list[int]:
        prefix = [DOMAIN_TOKEN_MAP["text"], 10, 11, 12, 13]
        body = [14, 15, 14, 16, 15, 17, 14, 15]
        seq = prefix + (body * ((self.seq_len // len(body)) + 1))
        return seq[: self.seq_len]

    def _make_retrieval_sequence(self) -> list[int]:
        key = self._rng.randint(20, 24)
        filler = [30, 31, 32, 33]
        seq = [DOMAIN_TOKEN_MAP["retrieval"], key, 34, 35]
        while len(seq) < self.seq_len:
            seq.extend(filler)
            seq.append(key)
            seq.append(36)
        return seq[: self.seq_len]

    def _make_code_sequence(self) -> list[int]:
        variants = [
            [DOMAIN_TOKEN_MAP["code"], 40, 41, 42, 43, 44, 45, 46],
            [DOMAIN_TOKEN_MAP["code"], 47, 48, 49, 50, 51, 52, 53],
        ]
        template = self._rng.choice(variants)
        seq = template + (template[1:] * ((self.seq_len // (len(template) - 1)) + 1))
        return seq[: self.seq_len]

    def make_sequence(self, domain: str) -> torch.Tensor:
        name = str(domain).lower()
        if name == "text":
            tokens = self._make_text_sequence()
        elif name == "retrieval":
            tokens = self._make_retrieval_sequence()
        elif name == "code":
            tokens = self._make_code_sequence()
        else:
            raise ValueError(f"unsupported domain: {domain}")
        return torch.tensor(tokens, dtype=torch.long, device=self.device_obj)

    def get_batch(self, batch_size: int, domain: str | None = None) -> tuple[torch.Tensor, list[str]]:
        domains: list[str] = []
        rows: list[torch.Tensor] = []
        for _ in range(batch_size):
            sample_domain = domain or self._rng.choice(DOMAIN_NAMES)
            domains.append(sample_domain)
            rows.append(self.make_sequence(sample_domain))
        batch = torch.stack(rows, dim=0)
        return batch, domains
