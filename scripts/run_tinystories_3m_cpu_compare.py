from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.the_stack_bpe import BPETokenDataset, _train_tokenizer
from src.fog.config import FOGConfig
from src.fog.model_baseline import BaselineTransformer
from src.fog.model_motif import MotifTransformer


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_tinystories_bpe_corpus(
    train_rows: int,
    val_rows: int,
    vocab_size: int,
    seq_len: int,
    device: str,
    cache_dir: Path,
) -> tuple[BPETokenDataset, BPETokenDataset, dict[str, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tinystories_rows{train_rows}_val{val_rows}_bpe{vocab_size}"
    tokenizer_path = cache_dir / f"{prefix}_tokenizer.json"
    train_ids_path = cache_dir / f"{prefix}_train_ids.pt"
    val_ids_path = cache_dir / f"{prefix}_val_ids.pt"
    meta_path = cache_dir / f"{prefix}_meta.json"

    if tokenizer_path.exists() and train_ids_path.exists() and val_ids_path.exists() and meta_path.exists():
        train_ids = torch.load(train_ids_path, map_location="cpu")
        val_ids = torch.load(val_ids_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{train_rows}]")
        val_ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{val_rows}]")

        train_text = "\n".join(example["text"] for example in train_ds)
        val_text = "\n".join(example["text"] for example in val_ds)

        tokenizer = _train_tokenizer(text=train_text, vocab_size=vocab_size)
        train_ids = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
        val_ids = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)
        actual_vocab_size = tokenizer.get_vocab_size()

        tokenizer.save(str(tokenizer_path))
        torch.save(train_ids, train_ids_path)
        torch.save(val_ids, val_ids_path)
        meta = {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "vocab_size": int(actual_vocab_size),
            "train_token_count": int(train_ids.numel()),
            "val_token_count": int(val_ids.numel()),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    train = BPETokenDataset(
        token_ids=train_ids,
        vocab_size=int(meta["vocab_size"]),
        split="train",
        seq_len=seq_len,
        device=device,
        split_data=False,
    )
    val = BPETokenDataset(
        token_ids=val_ids,
        vocab_size=int(meta["vocab_size"]),
        split="val",
        seq_len=seq_len,
        device=device,
        split_data=False,
    )
    return train, val, meta


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataset: BPETokenDataset,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for _ in range(eval_steps):
        x, y = dataset.get_batch(batch_size)
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)
        logits = out["logits"]
        total_loss += float(out["loss"].item())
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == y).sum().item())
        total_tokens += int(y.numel())

    return {
        "loss": total_loss / max(eval_steps, 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }


def train_steps(
    name: str,
    model: torch.nn.Module,
    train_data: BPETokenDataset,
    val_data: BPETokenDataset,
    batch_size: int,
    steps: int,
    lr: float,
    weight_decay: float,
    eval_every: int,
    eval_steps: int,
    device: torch.device,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    started_at = time.time()

    for step in range(1, steps + 1):
        model.train()
        x, y = train_data.get_batch(batch_size)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, y)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1 or step == steps:
            train_logits = out["logits"].detach()
            train_acc = float((train_logits.argmax(dim=-1) == y).float().mean().item())
            val_metrics = evaluate_model(
                model=model,
                dataset=val_data,
                batch_size=batch_size,
                eval_steps=eval_steps,
                device=device,
            )
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_accuracy": train_acc,
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
            history.append(row)
            print(
                f"[{name:7s}] step {step:>3d}/{steps} | "
                f"train_loss={row['train_loss']:.4f} | train_acc={row['train_accuracy']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | val_acc={row['val_accuracy']:.4f}"
            )

    elapsed = time.time() - started_at
    return {
        "params": count_params(model),
        "elapsed_s": elapsed,
        "history": history,
        "final": history[-1] if history else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyStories CPU compare: ~3M Uniform vs ~3M FOG")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--train_rows", type=int, default=5000)
    parser.add_argument("--val_rows", type=int, default=500)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=str,
        default="results/tinystories_3m_cpu_compare.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    device = torch.device(args.device)
    cache_dir = ROOT / "data_cache"
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_data, val_data, meta = build_tinystories_bpe_corpus(
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        device=args.device,
        cache_dir=cache_dir,
    )

    actual_vocab = train_data.vocab_size

    uniform_cfg = FOGConfig(
        vocab_size=actual_vocab,
        d_model=208,
        n_layers=5,
        n_heads=8,
        max_seq_len=args.seq_len,
        dropout=0.1,
        d_ff=832,
    )
    motif_cfg = FOGConfig(
        vocab_size=actual_vocab,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=args.seq_len,
        dropout=0.1,
        d_ff=1024,
        d_compare=64,
        d_memory=192,
        d_expand=512,
        d_gate=32,
    )

    uniform_model = BaselineTransformer(uniform_cfg).to(device)
    motif_model = MotifTransformer(motif_cfg).to(device)

    print(f"Device: {device}")
    print(
        "TinyStories BPE corpus | "
        f"train_rows={args.train_rows} | val_rows={args.val_rows} | "
        f"train_tokens={meta['train_token_count']:,} | val_tokens={meta['val_token_count']:,} | "
        f"vocab={actual_vocab}"
    )
    print(f"Uniform params: {count_params(uniform_model):,}")
    print(f"FOG params:     {count_params(motif_model):,}")

    uniform_result = train_steps(
        name="uniform",
        model=uniform_model,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        device=device,
    )
    motif_result = train_steps(
        name="fog",
        model=motif_model,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        device=device,
    )

    payload = {
        "dataset": {
            "name": "roneneldan/TinyStories",
            "train_rows": args.train_rows,
            "val_rows": args.val_rows,
            "seq_len": args.seq_len,
            "vocab_size": actual_vocab,
            "train_token_count": meta["train_token_count"],
            "val_token_count": meta["val_token_count"],
        },
        "runtime": {
            "device": args.device,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
            "eval_steps": args.eval_steps,
            "seed": args.seed,
        },
        "uniform": uniform_result,
        "fog": motif_result,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved results to: {output_path}")
    if uniform_result["final"] and motif_result["final"]:
        gap = float(motif_result["final"]["val_accuracy"]) - float(uniform_result["final"]["val_accuracy"])
        print(
            "Final summary | "
            f"Uniform val_acc={uniform_result['final']['val_accuracy']:.4f} | "
            f"FOG val_acc={motif_result['final']['val_accuracy']:.4f} | "
            f"Gap={gap:+.4f}"
        )


if __name__ == "__main__":
    main()
