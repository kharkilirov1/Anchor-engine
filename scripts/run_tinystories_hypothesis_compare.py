from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.the_stack_bpe import BPETokenDataset, _train_tokenizer
from src.data.the_stack import load_the_stack_text
from src.fog.config import FOGConfig
from src.fog.model_baseline import BaselineTransformer
from src.fog.model_code import (
    CodeAwareCopyTransformer,
    CodeAwareStructuredLightTransformer,
    CodeAwareStructuredTransformer,
    StructuredV2CopyTransformer,
)
from src.fog.model_fast import FastMotifTransformer, FastStructuredMotifTransformer
from src.fog.model_motif import MotifTransformer
from src.fog.model_runtime import RuntimeStructuredMotifTransformer
from src.fog.model_structured import StructuredMotifTransformer
from src.fog.model_structured_v2 import StructuredMotifTransformerV2


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
        if "train_byte_count" not in meta or "val_byte_count" not in meta:
            train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{train_rows}]")
            val_ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{val_rows}]")
            train_text = "\n".join(example["text"] for example in train_ds)
            val_text = "\n".join(example["text"] for example in val_ds)
            meta["train_byte_count"] = len(train_text.encode("utf-8"))
            meta["val_byte_count"] = len(val_text.encode("utf-8"))
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{train_rows}]")
        val_ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{val_rows}]")
        train_text = "\n".join(example["text"] for example in train_ds)
        val_text = "\n".join(example["text"] for example in val_ds)
        tokenizer = _train_tokenizer(text=train_text, vocab_size=vocab_size)
        train_ids = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
        val_ids = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)
        meta = {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "vocab_size": int(tokenizer.get_vocab_size()),
            "train_token_count": int(train_ids.numel()),
            "val_token_count": int(val_ids.numel()),
            "train_byte_count": len(train_text.encode("utf-8")),
            "val_byte_count": len(val_text.encode("utf-8")),
        }
        tokenizer.save(str(tokenizer_path))
        torch.save(train_ids, train_ids_path)
        torch.save(val_ids, val_ids_path)
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


def build_the_stack_bpe_corpus(
    target_bytes: int,
    vocab_size: int,
    seq_len: int,
    device: str,
    cache_dir: Path,
    repo_id: str,
    lang: str,
) -> tuple[BPETokenDataset, BPETokenDataset, dict[str, int | str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{repo_id.replace('/', '__').replace('-', '_')}_{lang}_{target_bytes}_bpe{vocab_size}"
    tokenizer_path = cache_dir / f"{prefix}_tokenizer.json"
    ids_path = cache_dir / f"{prefix}_ids.pt"
    meta_path = cache_dir / f"{prefix}_meta.json"

    if tokenizer_path.exists() and ids_path.exists() and meta_path.exists():
        token_ids = torch.load(ids_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        raw = load_the_stack_text(
            data_dir=str(cache_dir),
            repo_id=repo_id,
            lang=lang,
            target_bytes=target_bytes,
        )
        text = raw.decode("utf-8", errors="ignore")
        tokenizer = _train_tokenizer(text=text, vocab_size=vocab_size)
        token_ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
        meta = {
            "dataset": "the-stack-bpe",
            "repo_id": repo_id,
            "lang": lang,
            "target_bytes": target_bytes,
            "vocab_size": int(tokenizer.get_vocab_size()),
            "token_count": int(token_ids.numel()),
            "byte_count": len(raw),
        }
        tokenizer.save(str(tokenizer_path))
        torch.save(token_ids, ids_path)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    token_count = int(meta["token_count"])
    train_token_count = int(0.9 * token_count)
    val_token_count = token_count - train_token_count
    byte_count = int(meta["byte_count"])
    train_byte_count = int(0.9 * byte_count)
    val_byte_count = byte_count - train_byte_count

    train = BPETokenDataset(
        token_ids=token_ids,
        vocab_size=int(meta["vocab_size"]),
        split="train",
        seq_len=seq_len,
        device=device,
        split_data=True,
    )
    val = BPETokenDataset(
        token_ids=token_ids,
        vocab_size=int(meta["vocab_size"]),
        split="val",
        seq_len=seq_len,
        device=device,
        split_data=True,
    )
    return train, val, {
        "repo_id": str(meta["repo_id"]),
        "lang": str(meta["lang"]),
        "vocab_size": int(meta["vocab_size"]),
        "train_token_count": train_token_count,
        "val_token_count": val_token_count,
        "train_byte_count": train_byte_count,
        "val_byte_count": val_byte_count,
        "target_bytes": int(meta["target_bytes"]),
    }


def build_optimizer(
    model: torch.nn.Module,
    recipe: str,
    lr: float,
    weight_decay: float,
    steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR | None]:
    if recipe == "plain":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay), None

    interface_params: list[torch.nn.Parameter] = []
    control_params: list[torch.nn.Parameter] = []
    core_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in ("tok_emb", "pos_emb", "head", "ln")):
            interface_params.append(param)
        elif any(token in name for token in ("gate", "scale")):
            control_params.append(param)
        else:
            core_params.append(param)

    if recipe == "structured_stable":
        param_groups = [
            {"params": interface_params, "lr": lr * 0.55, "weight_decay": weight_decay},
            {"params": core_params, "lr": lr * 0.70, "weight_decay": weight_decay * 0.8},
            {"params": control_params, "lr": lr * 0.95, "weight_decay": weight_decay * 0.25},
        ]
    else:
        param_groups = [
            {"params": interface_params, "lr": lr * 0.75, "weight_decay": weight_decay},
            {"params": core_params, "lr": lr, "weight_decay": weight_decay},
            {"params": control_params, "lr": lr * 1.35, "weight_decay": weight_decay * 0.5},
        ]
    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = max(5, steps // (5 if recipe == "structured_stable" else 10))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.2 + 0.8 * ((step + 1) / warmup_steps)
        if recipe == "structured_stable":
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.35 + 0.65 * cosine
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def loss_to_bpt(loss_nats: float) -> float:
    return loss_nats / math.log(2.0)


def loss_to_ppl(loss_nats: float) -> float:
    return math.exp(min(loss_nats, 20.0))


def approximate_bpb(loss_nats: float, token_count: int, byte_count: int) -> float:
    if byte_count <= 0:
        return float("nan")
    return loss_to_bpt(loss_nats) * (token_count / byte_count)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataset: BPETokenDataset,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
    byte_count: int,
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
        total_loss += float(out["loss"].item())
        preds = out["logits"].argmax(dim=-1)
        total_correct += int((preds == y).sum().item())
        total_tokens += int(y.numel())

    loss = total_loss / max(eval_steps, 1)
    return {
        "loss": loss,
        "accuracy": total_correct / max(total_tokens, 1),
        "bpt": loss_to_bpt(loss),
        "ppl": loss_to_ppl(loss),
        "approx_bpb": approximate_bpb(loss, len(dataset), byte_count),
        "tokens_evaluated": total_tokens,
    }


def train_model(
    name: str,
    model: torch.nn.Module,
    recipe: str,
    train_data: BPETokenDataset,
    val_data: BPETokenDataset,
    batch_size: int,
    steps: int,
    lr: float,
    weight_decay: float,
    eval_every: int,
    eval_steps: int,
    device: torch.device,
    train_byte_count: int,
    val_byte_count: int,
    time_budget_s: float | None = None,
) -> dict[str, object]:
    optimizer, scheduler = build_optimizer(
        model=model,
        recipe=recipe,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
    )
    history: list[dict[str, float | int]] = []
    started_at = time.time()
    train_tokens_processed = 0

    last_step = 0
    for step in range(1, steps + 1):
        model.train()
        x, y = train_data.get_batch(batch_size)
        x = x.to(device)
        y = y.to(device)
        train_tokens_processed += int(y.numel())

        optimizer.zero_grad(set_to_none=True)
        out = model(x, y)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % eval_every == 0 or step == 1 or step == steps:
            train_acc = float((out["logits"].argmax(dim=-1) == y).float().mean().item())
            val_metrics = evaluate_model(
                model=model,
                dataset=val_data,
                batch_size=batch_size,
                eval_steps=eval_steps,
                device=device,
                byte_count=val_byte_count,
            )
            elapsed = time.time() - started_at
            train_loss = float(loss.item())
            row = {
                "step": step,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_bpt": loss_to_bpt(train_loss),
                "train_ppl": loss_to_ppl(train_loss),
                "train_approx_bpb": approximate_bpb(train_loss, len(train_data), train_byte_count),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_bpt": float(val_metrics["bpt"]),
                "val_ppl": float(val_metrics["ppl"]),
                "val_approx_bpb": float(val_metrics["approx_bpb"]),
                "lr_group0": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": elapsed,
                "steps_per_s": step / max(elapsed, 1e-9),
                "train_tokens_per_s": train_tokens_processed / max(elapsed, 1e-9),
            }
            history.append(row)
            last_step = step
            print(
                f"[{name:16s}] step {step:>3d}/{steps} | "
                f"train_loss={row['train_loss']:.4f} | train_bpt={row['train_bpt']:.3f} | "
                f"val_loss={row['val_loss']:.4f} | val_bpt={row['val_bpt']:.3f} | "
                f"val_ppl={row['val_ppl']:.2f} | val_acc={row['val_accuracy']:.4f} | "
                f"tok/s={row['train_tokens_per_s']:.0f} | lr0={row['lr_group0']:.6f}"
            )

        if time_budget_s is not None and (time.time() - started_at) >= time_budget_s:
            if last_step != step:
                train_acc = float((out["logits"].argmax(dim=-1) == y).float().mean().item())
                val_metrics = evaluate_model(
                    model=model,
                    dataset=val_data,
                    batch_size=batch_size,
                    eval_steps=eval_steps,
                    device=device,
                    byte_count=val_byte_count,
                )
                elapsed = time.time() - started_at
                train_loss = float(loss.item())
                history.append(
                    {
                        "step": step,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "train_bpt": loss_to_bpt(train_loss),
                        "train_ppl": loss_to_ppl(train_loss),
                        "train_approx_bpb": approximate_bpb(train_loss, len(train_data), train_byte_count),
                        "val_loss": float(val_metrics["loss"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "val_bpt": float(val_metrics["bpt"]),
                        "val_ppl": float(val_metrics["ppl"]),
                        "val_approx_bpb": float(val_metrics["approx_bpb"]),
                        "lr_group0": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_s": elapsed,
                        "steps_per_s": step / max(elapsed, 1e-9),
                        "train_tokens_per_s": train_tokens_processed / max(elapsed, 1e-9),
                    }
                )
            break

    elapsed_total = time.time() - started_at
    result: dict[str, object] = {
        "params": count_params(model),
        "recipe": recipe,
        "elapsed_s": elapsed_total,
        "steps_completed": history[-1]["step"] if history else 0,
        "tokens_processed": train_tokens_processed,
        "history": history,
        "final": history[-1] if history else {},
        "best_by_val_loss": min(history, key=lambda row: row["val_loss"]) if history else {},
        "best_by_val_acc": max(history, key=lambda row: row["val_accuracy"]) if history else {},
    }
    if hasattr(model, "layer_geometries"):
        result["geometry"] = [
            {
                "stage": g.stage,
                "d_compare": g.d_compare,
                "d_memory": g.d_memory,
                "d_expand": g.d_expand,
                "d_gate": g.d_gate,
            }
            for g in model.layer_geometries
        ]
    return result


def build_summary_tables(results: dict[str, object], model_names: list[str]) -> tuple[str, str]:
    summary_lines = [
        "| Model | Params | Recipe | Steps Done | Best Loss Step | Best Val Loss | Best Val BPT | Best Val PPL | Best Acc Step | Best Val Acc | Best Approx BPB | Final Val Loss | Final Val Acc | Steps/s | Train tok/s | Elapsed s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    curve_lines = [
        "| Model | Step | Train Loss | Train BPT | Train Approx BPB | Val Loss | Val BPT | Val PPL | Val Acc | Val Approx BPB | Steps/s | Train tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in model_names:
        payload = results[name]
        best_loss = payload["best_by_val_loss"]
        best_acc = payload["best_by_val_acc"]
        final = payload["final"]
        summary_lines.append(
            f"| {name} | {payload['params']:,} | {payload['recipe']} | {payload.get('steps_completed', 0)} | {best_loss['step']} | "
            f"{best_loss['val_loss']:.4f} | {best_loss['val_bpt']:.3f} | {best_loss['val_ppl']:.2f} | "
            f"{best_acc['step']} | {best_acc['val_accuracy']:.4f} | {best_loss['val_approx_bpb']:.3f} | "
            f"{final['val_loss']:.4f} | {final['val_accuracy']:.4f} | "
            f"{final['steps_per_s']:.2f} | {final['train_tokens_per_s']:.0f} | {payload['elapsed_s']:.2f} |"
        )
        for row in payload["history"]:
            curve_lines.append(
                f"| {name} | {row['step']} | {row['train_loss']:.4f} | {row['train_bpt']:.3f} | "
                f"{row['train_approx_bpb']:.3f} | {row['val_loss']:.4f} | {row['val_bpt']:.3f} | "
                f"{row['val_ppl']:.2f} | {row['val_accuracy']:.4f} | {row['val_approx_bpb']:.3f} | "
                f"{row['steps_per_s']:.2f} | {row['train_tokens_per_s']:.0f} |"
            )

    return "\n".join(summary_lines), "\n".join(curve_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyStories FOG hypothesis suite")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "code"])
    parser.add_argument("--train_rows", type=int, default=1000)
    parser.add_argument("--val_rows", type=int, default=100)
    parser.add_argument("--code_repo", type=str, default="bigcode/the-stack-smol-xs")
    parser.add_argument("--code_lang", type=str, default="python")
    parser.add_argument("--code_bytes", type=int, default=1_200_000)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--time_budget_s", type=float, default=0.0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "uniform",
            "motif",
            "structured",
            "structured_v2",
            "structured_runtime",
        ],
        choices=[
            "uniform",
            "motif",
            "motif_fast",
            "structured",
            "structured_copy",
            "structured_v2_copy",
            "structured_code",
            "structured_code_light",
            "structured_v2",
            "structured_runtime",
            "structured_fast",
            "structured_easy",
            "structured_fast_easy",
        ],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tinystories_hypothesis_compare.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = torch.device(args.device)

    if args.dataset == "tinystories":
        train_data, val_data, meta = build_tinystories_bpe_corpus(
            train_rows=args.train_rows,
            val_rows=args.val_rows,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            device=args.device,
            cache_dir=ROOT / "data_cache",
        )
        dataset_meta = {
            "name": "roneneldan/TinyStories",
            "train_rows": args.train_rows,
            "val_rows": args.val_rows,
            "seq_len": args.seq_len,
            "vocab_size": meta["vocab_size"],
            "train_token_count": meta["train_token_count"],
            "val_token_count": meta["val_token_count"],
        }
    else:
        train_data, val_data, meta = build_the_stack_bpe_corpus(
            target_bytes=args.code_bytes,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            device=args.device,
            cache_dir=ROOT / "data_cache",
            repo_id=args.code_repo,
            lang=args.code_lang,
        )
        dataset_meta = {
            "name": "the-stack-bpe",
            "repo_id": args.code_repo,
            "lang": args.code_lang,
            "target_bytes": args.code_bytes,
            "seq_len": args.seq_len,
            "vocab_size": meta["vocab_size"],
            "train_token_count": meta["train_token_count"],
            "val_token_count": meta["val_token_count"],
        }
    vocab_size = train_data.vocab_size

    uniform_cfg = FOGConfig(
        vocab_size=vocab_size,
        d_model=208,
        n_layers=5,
        n_heads=8,
        max_seq_len=args.seq_len,
        dropout=0.1,
        d_ff=832,
    )
    motif_cfg = FOGConfig(
        vocab_size=vocab_size,
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
    structured_cfg = FOGConfig(
        vocab_size=vocab_size,
        d_model=224,
        n_layers=5,
        n_heads=8,
        max_seq_len=args.seq_len,
        dropout=0.1,
        d_ff=896,
        d_compare=56,
        d_memory=168,
        d_expand=448,
        d_gate=28,
    )

    model_builders = {
        "uniform": lambda: (BaselineTransformer(uniform_cfg).to(device), "plain"),
        "motif": lambda: (MotifTransformer(motif_cfg).to(device), "plain"),
        "motif_fast": lambda: (FastMotifTransformer(motif_cfg).to(device), "plain"),
        "structured": lambda: (StructuredMotifTransformer(structured_cfg).to(device), "plain"),
        "structured_copy": lambda: (CodeAwareCopyTransformer(structured_cfg).to(device), "plain"),
        "structured_v2_copy": lambda: (StructuredV2CopyTransformer(structured_cfg).to(device), "plain"),
        "structured_code": lambda: (CodeAwareStructuredTransformer(structured_cfg).to(device), "plain"),
        "structured_code_light": lambda: (CodeAwareStructuredLightTransformer(structured_cfg).to(device), "plain"),
        "structured_v2": lambda: (StructuredMotifTransformerV2(structured_cfg).to(device), "plain"),
        "structured_runtime": lambda: (RuntimeStructuredMotifTransformer(structured_cfg).to(device), "plain"),
        "structured_fast": lambda: (FastStructuredMotifTransformer(structured_cfg).to(device), "plain"),
        "structured_easy": lambda: (StructuredMotifTransformer(structured_cfg).to(device), "structured_stable"),
        "structured_fast_easy": lambda: (FastStructuredMotifTransformer(structured_cfg).to(device), "structured_stable"),
    }

    print(f"Device: {device}")
    if args.dataset == "tinystories":
        print(
            "TinyStories hypothesis suite | "
            f"train_rows={args.train_rows} | val_rows={args.val_rows} | "
            f"train_tokens={meta['train_token_count']:,} | val_tokens={meta['val_token_count']:,} | "
            f"vocab={vocab_size}"
        )
    else:
        print(
            "Code hypothesis suite | "
            f"repo={args.code_repo} | lang={args.code_lang} | bytes={args.code_bytes:,} | "
            f"train_tokens={meta['train_token_count']:,} | val_tokens={meta['val_token_count']:,} | "
            f"vocab={vocab_size}"
        )

    results: dict[str, object] = {
        "dataset": dataset_meta,
        "runtime": {
            "device": args.device,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
            "eval_steps": args.eval_steps,
            "time_budget_s": args.time_budget_s,
            "seed": args.seed,
            "models": args.models,
        },
    }

    for name in args.models:
        model, recipe = model_builders[name]()
        print(f"{name:16s} params={count_params(model):,} recipe={recipe}")
        results[name] = train_model(
            name=name,
            model=model,
            recipe=recipe,
            train_data=train_data,
            val_data=val_data,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            eval_steps=args.eval_steps,
            device=device,
            train_byte_count=meta["train_byte_count"],
            val_byte_count=meta["val_byte_count"],
            time_budget_s=args.time_budget_s if args.time_budget_s > 0 else None,
        )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to: {output_path}")

    summary_table, curve_table = build_summary_tables(results, args.models)
    summary_path = output_path.with_suffix(".summary.md")
    curves_path = output_path.with_suffix(".curves.md")
    summary_path.write_text(summary_table + "\n", encoding="utf-8")
    curves_path.write_text(curve_table + "\n", encoding="utf-8")
    print("\n=== SUMMARY TABLE ===")
    print(summary_table)
    print(f"\nSaved summary table to: {summary_path}")
    print(f"Saved curves table to: {curves_path}")


if __name__ == "__main__":
    main()
