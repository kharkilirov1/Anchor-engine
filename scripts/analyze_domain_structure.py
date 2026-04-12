from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.anchor_synthetic import AnchorSyntheticDataset
from src.data.tinystories_bpe import load_tinystories_bpe
from src.utils.domain_structure import (
    compute_sequence_structure_stats,
    compute_token_structure_stats,
    stats_to_dict,
)


def _anchor_sequences(seq_len: int) -> torch.Tensor:
    dataset = AnchorSyntheticDataset(split="train", seq_len=seq_len, device="cpu")
    return torch.stack([sample[0] for sample in dataset.samples], dim=0)


def _tinystories_sequences(
    seq_len: int,
    max_sequences: int,
    data_dir: str,
    target_bytes: int,
    vocab_size: int,
) -> tuple[torch.Tensor, int]:
    train, _ = load_tinystories_bpe(
        seq_len=seq_len,
        device="cpu",
        data_dir=data_dir,
        target_bytes=target_bytes,
        vocab_size=vocab_size,
    )
    token_ids = train.data.cpu()
    max_start = max(0, token_ids.numel() - seq_len)
    if max_start == 0:
        raise ValueError("TinyStories cache is too small for the requested seq_len.")
    stride = max(1, max_start // max_sequences)
    starts = list(range(0, max_start, stride))[:max_sequences]
    windows = torch.stack([token_ids[s : s + seq_len] for s in starts], dim=0)
    return windows, train.vocab_size


def _format_float(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze domain structure for anchor-synthetic vs TinyStories.")
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "domain_structure_v1"))
    parser.add_argument("--data_dir", default=str(ROOT / "data_cache"))
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--max_sequences", type=int, default=576)
    parser.add_argument("--tinystories_bytes", type=int, default=180_000)
    parser.add_argument("--tinystories_vocab_size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anchor_sequences = _anchor_sequences(args.seq_len)
    tiny_sequences, tiny_vocab = _tinystories_sequences(
        seq_len=args.seq_len,
        max_sequences=args.max_sequences,
        data_dir=args.data_dir,
        target_bytes=args.tinystories_bytes,
        vocab_size=args.tinystories_vocab_size,
    )

    report = {
        "anchor_synthetic": {
            "token": stats_to_dict(compute_token_structure_stats(anchor_sequences.flatten())),
            "sequence": stats_to_dict(compute_sequence_structure_stats(anchor_sequences)),
        },
        "tinystories_bpe": {
            "token": stats_to_dict(compute_token_structure_stats(tiny_sequences.flatten(), vocab_size=tiny_vocab)),
            "sequence": stats_to_dict(compute_sequence_structure_stats(tiny_sequences)),
        },
        "analysis_config": {
            "seq_len": args.seq_len,
            "max_sequences": args.max_sequences,
            "tinystories_bytes": args.tinystories_bytes,
            "tinystories_vocab_size": args.tinystories_vocab_size,
        },
    }

    json_path = output_dir / "domain_structure.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Domain structure analysis", ""]
    for scope in ("token", "sequence"):
        lines.append(f"## {scope.capitalize()} statistics")
        lines.append("")
        lines.append("| Metric | anchor-synthetic | tinystories-bpe |")
        lines.append("|---|---:|---:|")
        keys = report["anchor_synthetic"][scope].keys()
        for key in keys:
            left = report["anchor_synthetic"][scope][key]
            right = report["tinystories_bpe"][scope][key]
            lines.append(f"| {key} | {_format_float(left)} | {_format_float(right)} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- anchor-synthetic is expected to have far smaller effective support, heavier template reuse, and lower conditional entropy.")
    lines.append("- tinystories-bpe should look more diverse at both token and sequence levels, even with the same analysis window.")
    lines.append("- If an architecture wins on anchor-synthetic but not on TinyStories, the likely cause is not \"magic domain dependence\" but inductive-bias matching: the model is better aligned to repeated symbolic templates than to broad natural-text continuation.")
    lines.append("")

    md_path = output_dir / "summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
