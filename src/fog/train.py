"""Train and compare baseline vs motif-aware transformer on algorithmic tasks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.fog.config import (
    FOGConfig,
    BASELINE_SMALL, MOTIF_SMALL,
    BASELINE_TINY, MOTIF_TINY, UNIFORM_TINY,
    BASELINE_MICRO, MOTIF_MICRO, UNIFORM_MICRO,
    BASELINE_MED, MOTIF_MED, UNIFORM_MED,
)
from src.fog.model_baseline import BaselineTransformer
from src.fog.model_motif import MotifTransformer
from src.fog.data import (
    CopyTask, ReverseTask, SelectiveRetrieval,
    DistractorRetrieval, NoisyRetrieval, MultiQueryRetrieval,
    ChainedRetrieval,
    DenseRetrieval, NoisyDenseRetrieval, SortedRetrieval, MultiHopChained,
    prebatch_dataset, TensorBatchIterator,
)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_epoch(
    model: torch.nn.Module,
    loader: TensorBatchIterator,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        out = model(input_ids, targets, loss_mask=loss_mask)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_accuracy(
    model: torch.nn.Module,
    loader: TensorBatchIterator,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    seq_correct = 0
    seq_total = 0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        out = model(input_ids, targets, loss_mask=loss_mask)
        total_loss += out["loss"].item()
        n_batches += 1

        preds = out["logits"].argmax(dim=-1)
        m = loss_mask.bool()
        correct += (preds[m] == targets[m]).sum().item()
        total += m.sum().item()
        for b in range(preds.size(0)):
            mb = m[b]
            if mb.any():
                seq_total += 1
                if torch.equal(preds[b][mb], targets[b][mb]):
                    seq_correct += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
        "exact_match": seq_correct / max(seq_total, 1),
        "total_tokens": total,
    }


TASK_MAP = {
    "copy": CopyTask,
    "reverse": ReverseTask,
    "retrieval": SelectiveRetrieval,
    "distractor": DistractorRetrieval,
    "noisy": NoisyRetrieval,
    "multiquery": MultiQueryRetrieval,
    "chained": ChainedRetrieval,
    "dense": DenseRetrieval,
    "noisy_dense": NoisyDenseRetrieval,
    "sorted": SortedRetrieval,
    "multihop": MultiHopChained,
}


def run_experiment(
    task_name: str,
    cfg: FOGConfig,
    model_type: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
    n_train: int = 2000,
    n_eval: int = 500,
) -> dict:
    torch.manual_seed(seed)

    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASK_MAP.keys())}")
    task_cls = TASK_MAP[task_name]

    # Use n_pairs=6 for chained (needs enough pairs for chains to form)
    extra_kwargs = {}
    if task_name == "chained":
        extra_kwargs["n_pairs"] = 6
    elif task_name == "multihop":
        extra_kwargs["n_pairs"] = 12
    elif task_name == "dense":
        extra_kwargs["n_pairs"] = 16
    elif task_name == "noisy_dense":
        extra_kwargs["n_pairs"] = 10
        extra_kwargs["noise_len"] = 4
    elif task_name == "sorted":
        extra_kwargs["n_pairs"] = 6
    elif task_name in ("distractor", "noisy", "multiquery", "retrieval"):
        extra_kwargs["n_pairs"] = 4

    train_ds = task_cls(cfg.vocab_size, cfg.max_seq_len, n_train, seed=0, **extra_kwargs)
    eval_ds = task_cls(cfg.vocab_size, cfg.max_seq_len, n_eval, seed=99, **extra_kwargs)

    # Pre-batch into contiguous tensors for speed
    train_data = prebatch_dataset(train_ds, cfg.max_seq_len)
    eval_data = prebatch_dataset(eval_ds, cfg.max_seq_len)
    train_loader = TensorBatchIterator(train_data, batch_size, shuffle=True)
    eval_loader = TensorBatchIterator(eval_data, batch_size, shuffle=False)

    if model_type in ("baseline", "uniform_small"):
        model = BaselineTransformer(cfg).to(device)
    elif model_type == "motif":
        model = MotifTransformer(cfg).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = eval_accuracy(model, eval_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "eval_loss": round(metrics["loss"], 4),
            "eval_accuracy": round(metrics["accuracy"], 4),
            "eval_exact_match": round(metrics["exact_match"], 4),
        })
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_type}/{task_name}] epoch {epoch:>3d}  "
                  f"train={train_loss:.4f}  eval={metrics['loss']:.4f}  "
                  f"acc={metrics['accuracy']:.4f}  em={metrics['exact_match']:.4f}")

    elapsed = time.time() - t0
    final = history[-1] if history else {}
    return {
        "model_type": model_type,
        "task": task_name,
        "seed": seed,
        "n_params": n_params,
        "n_epochs": n_epochs,
        "elapsed_s": round(elapsed, 1),
        "final_train_loss": final.get("train_loss"),
        "final_eval_loss": final.get("eval_loss"),
        "final_accuracy": final.get("eval_accuracy"),
        "final_exact_match": final.get("eval_exact_match"),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FOG Ablation: baseline vs motif-aware")
    parser.add_argument("--tasks", nargs="+", default=["copy", "reverse", "retrieval"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--size", type=str, default="med",
                        choices=["micro", "tiny", "med", "small"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--output", type=str, default="archive/fog_ablation.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.size == "micro":
        configs = [
            ("baseline", BASELINE_MICRO),
            ("uniform_small", UNIFORM_MICRO),
            ("motif", MOTIF_MICRO),
        ]
    elif args.size == "tiny":
        configs = [
            ("baseline", BASELINE_TINY),
            ("uniform_small", UNIFORM_TINY),
            ("motif", MOTIF_TINY),
        ]
    elif args.size == "med":
        configs = [
            ("baseline", BASELINE_MED),
            ("uniform_small", UNIFORM_MED),
            ("motif", MOTIF_MED),
        ]
    else:
        configs = [("baseline", BASELINE_SMALL), ("motif", MOTIF_SMALL)]

    results = []

    for task in args.tasks:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"  Task: {task} (size={args.size}, seed={seed})")
            print(f"{'='*60}")

            for model_type, cfg in configs:
                result = run_experiment(
                    task_name=task,
                    cfg=cfg,
                    model_type=model_type,
                    n_epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    seed=seed,
                    n_train=args.n_train,
                    n_eval=args.n_eval,
                )
                results.append(result)
                print(f"  -> {model_type}: params={result['n_params']:,}  "
                      f"acc={result['final_accuracy']:.4f}  "
                      f"em={result['final_exact_match']:.4f}  "
                      f"time={result['elapsed_s']}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<12} {'Model':<15} {'Params':>8} {'Loss':>8} {'Acc':>8} {'EM':>8} {'Time':>6}")
    print("-" * 70)
    for r in results:
        em = r.get('final_exact_match', 0) or 0
        print(f"{r['task']:<12} {r['model_type']:<15} {r['n_params']:>8,} "
              f"{r['final_eval_loss']:>8.4f} {r['final_accuracy']:>8.4f} "
              f"{em:>8.4f} {r['elapsed_s']:>5.0f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
