import argparse
from collections import defaultdict

import torch

from src.model.config import PRESETS, ModelConfig
from train import _init_data, build_model, get_batch
from src.utils.metrics import bits_per_byte, branch_diversity


def evaluate(
    cfg: ModelConfig,
    device: str = "cpu",
    stage: str = "a",
    data_dir: str = "data_cache",
    dataset: str = "shakespeare",
    num_batches: int = 4,
    the_stack_repo: str = "bigcode/the-stack-smol-xs",
    the_stack_lang: str = "python",
    the_stack_bytes: int = 8_000_000,
    the_stack_vocab_size: int = 4096,
    tinystories_repo: str = "roneneldan/TinyStories",
    tinystories_bytes: int = 16_000_000,
    tinystories_vocab_size: int = 4096,
) -> dict[str, float]:
    _init_data(
        cfg,
        device,
        data_dir,
        dataset,
        the_stack_repo=the_stack_repo,
        the_stack_lang=the_stack_lang,
        the_stack_bytes=the_stack_bytes,
        the_stack_vocab_size=the_stack_vocab_size,
        tinystories_repo=tinystories_repo,
        tinystories_bytes=tinystories_bytes,
        tinystories_vocab_size=tinystories_vocab_size,
    )
    model = build_model(cfg, stage, device)
    model.eval()

    totals: dict[str, float] = defaultdict(float)
    branch_diversity_count = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(cfg, device, split="val")
            out = model(x, y)
            totals["loss"] += float(out["loss"].item())
            totals["ce_loss"] += float(out["ce_loss"].item())

            if "diversity_loss" in out:
                totals["diversity_loss"] += float(out["diversity_loss"].item())
            if "confidence" in out:
                totals["confidence"] += float(out["confidence"].mean().item())
            if "branch_logits" in out:
                totals["branch_diversity"] += float(branch_diversity(out["branch_logits"]))
                branch_diversity_count += 1
            if "anchor_diagnostics" in out:
                diag = out["anchor_diagnostics"]
                totals["anchors_active"] += float(diag["num_active"])
                totals["anchor_contradiction"] += float(diag["mean_contradiction_pressure"])
                totals["anchor_viability"] += float(diag["mean_viability"])
                totals["anchor_dead_end"] += float(diag["dead_end_count"])
            if "proposal_diagnostics" in out:
                diag = out["proposal_diagnostics"]
                totals["proposal_influence"] += float(diag["anchors_with_proposal_influence"])
                totals["proposal_blend"] += float(diag["mean_blend_ratio"])
                totals["strong_retire_gap"] += float(diag["mean_strong_retire_gap"])
            if "component_losses" in out:
                for name, value in out["component_losses"].items():
                    if name == "ce_loss":
                        continue
                    totals[f"component_{name}"] += float(value.item())

    metrics = {
        "loss": totals["loss"] / num_batches,
        "ce_loss": totals["ce_loss"] / num_batches,
        "bpb": bits_per_byte(totals["ce_loss"] / num_batches),
    }

    if "diversity_loss" in totals:
        metrics["diversity_loss"] = totals["diversity_loss"] / num_batches
    if "confidence" in totals:
        metrics["confidence"] = totals["confidence"] / num_batches
    if branch_diversity_count > 0:
        metrics["branch_diversity"] = totals["branch_diversity"] / branch_diversity_count
    if "anchors_active" in totals:
        metrics["anchors_active"] = totals["anchors_active"] / num_batches
        metrics["anchor_contradiction"] = totals["anchor_contradiction"] / num_batches
        metrics["anchor_viability"] = totals["anchor_viability"] / num_batches
        metrics["anchor_dead_end"] = totals["anchor_dead_end"] / num_batches
    if "proposal_influence" in totals:
        metrics["proposal_influence"] = totals["proposal_influence"] / num_batches
        metrics["proposal_blend"] = totals["proposal_blend"] / num_batches
        metrics["strong_retire_gap"] = totals["strong_retire_gap"] / num_batches
    for name, value in totals.items():
        if not name.startswith("component_"):
            continue
        metrics[name] = value / num_batches

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABPT Evaluation")
    parser.add_argument("--preset", default="toy", choices=list(PRESETS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage", default="a", choices=["a", "b", "anchor"])
    parser.add_argument("--data_dir", default="data_cache")
    parser.add_argument("--dataset", default="shakespeare", choices=["shakespeare", "anchor-synthetic", "the-stack", "the-stack-bpe", "tinystories-bpe"])
    parser.add_argument("--the_stack_repo", default="bigcode/the-stack-smol-xs")
    parser.add_argument("--the_stack_lang", default="python")
    parser.add_argument("--the_stack_bytes", type=int, default=8_000_000)
    parser.add_argument("--the_stack_vocab_size", type=int, default=4096)
    parser.add_argument("--tinystories_repo", default="roneneldan/TinyStories")
    parser.add_argument("--tinystories_bytes", type=int, default=16_000_000)
    parser.add_argument("--tinystories_vocab_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len
    metrics = evaluate(
        cfg=cfg,
        device=args.device,
        stage=args.stage,
        data_dir=args.data_dir,
        dataset=args.dataset,
        num_batches=args.batches,
        the_stack_repo=args.the_stack_repo,
        the_stack_lang=args.the_stack_lang,
        the_stack_bytes=args.the_stack_bytes,
        the_stack_vocab_size=args.the_stack_vocab_size,
        tinystories_repo=args.tinystories_repo,
        tinystories_bytes=args.tinystories_bytes,
        tinystories_vocab_size=args.tinystories_vocab_size,
    )
    print(metrics)
