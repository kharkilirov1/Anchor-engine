from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FOGConfig:
    # shared
    vocab_size: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1

    # baseline FFN
    d_ff: int = 1024

    # motif-aware attention
    d_compare: int = 64
    d_memory: int = 192

    # motif-aware FFN
    d_expand: int = 512
    d_gate: int = 32


BASELINE_SMALL = FOGConfig()

MOTIF_SMALL = FOGConfig(
    d_compare=64,
    d_memory=192,
    d_expand=512,
    d_gate=32,
)

# Param-matched uniform baseline for controlled comparison
# d_model=94, d_ff=376 → ~432K params to match MOTIF_TINY
UNIFORM_TINY = FOGConfig(
    vocab_size=32,
    d_model=94,
    n_layers=4,
    n_heads=2,
    max_seq_len=32,
    d_ff=376,
)

# Tiny configs for fast iteration
BASELINE_TINY = FOGConfig(
    vocab_size=32,
    d_model=128,
    n_layers=4,
    n_heads=4,
    max_seq_len=32,
    d_ff=512,
)

MOTIF_TINY = FOGConfig(
    vocab_size=32,
    d_model=128,
    n_layers=4,
    n_heads=4,
    max_seq_len=32,
    d_ff=512,
    d_compare=32,
    d_memory=96,
    d_expand=256,
    d_gate=16,
)
