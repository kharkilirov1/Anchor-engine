import argparse
import json
import math
import os
import torch
import torch.nn as nn
from src.model.config import PRESETS, ModelConfig
from src.model.abpt import ABPTModel
from src.model.abpt_b import ABPTModelB
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.utils.metrics import bits_per_byte, branch_diversity
from src.data.shakespeare import load_shakespeare, ShakespeareDataset
from src.data.anchor_synthetic import load_anchor_synthetic, AnchorSyntheticDataset
from src.data.the_stack import load_the_stack, ByteCorpusDataset
from src.data.the_stack_bpe import load_the_stack_bpe, BPETokenDataset
from src.data.tinystories_bpe import load_tinystories_bpe


_train_data: ShakespeareDataset | AnchorSyntheticDataset | ByteCorpusDataset | BPETokenDataset | None = None
_val_data: ShakespeareDataset | AnchorSyntheticDataset | ByteCorpusDataset | BPETokenDataset | None = None


def _init_data(
    cfg: ModelConfig,
    device: str,
    data_dir: str = "data_cache",
    dataset: str = "shakespeare",
    the_stack_repo: str = "bigcode/the-stack-smol-xs",
    the_stack_lang: str = "python",
    the_stack_bytes: int = 8_000_000,
    the_stack_vocab_size: int = 4096,
    tinystories_repo: str = "roneneldan/TinyStories",
    tinystories_bytes: int = 16_000_000,
    tinystories_vocab_size: int = 4096,
):
    """Load requested dataset, else fall back to random."""
    global _train_data, _val_data
    if dataset == "anchor-synthetic":
        _train_data, _val_data = load_anchor_synthetic(seq_len=24, device=device)
        cfg.max_seq_len = 24
        cfg.vocab_size = max(cfg.vocab_size, _train_data.vocab_size)
        print(
            f"Data: anchor-synthetic | vocab={_train_data.vocab_size} "
            f"| train={len(_train_data):,} | val={len(_val_data):,} samples"
        )
        return

    if dataset == "the-stack":
        _train_data, _val_data = load_the_stack(
            seq_len=cfg.max_seq_len,
            device=device,
            data_dir=data_dir,
            repo_id=the_stack_repo,
            lang=the_stack_lang,
            target_bytes=the_stack_bytes,
        )
        cfg.vocab_size = _train_data.vocab_size
        print(
            f"Data: the-stack | repo={the_stack_repo} | lang={the_stack_lang} "
            f"| vocab={_train_data.vocab_size} | train={len(_train_data):,} "
            f"| val={len(_val_data):,} bytes"
        )
        return

    if dataset == "the-stack-bpe":
        _train_data, _val_data = load_the_stack_bpe(
            seq_len=cfg.max_seq_len,
            device=device,
            data_dir=data_dir,
            repo_id=the_stack_repo,
            lang=the_stack_lang,
            target_bytes=the_stack_bytes,
            vocab_size=the_stack_vocab_size,
        )
        cfg.vocab_size = _train_data.vocab_size
        print(
            f"Data: the-stack-bpe | repo={the_stack_repo} | lang={the_stack_lang} "
            f"| vocab={_train_data.vocab_size} | train={len(_train_data):,} "
            f"| val={len(_val_data):,} tokens"
        )
        return

    if dataset == "tinystories-bpe":
        _train_data, _val_data = load_tinystories_bpe(
            seq_len=cfg.max_seq_len,
            device=device,
            data_dir=data_dir,
            repo_id=tinystories_repo,
            target_bytes=tinystories_bytes,
            vocab_size=tinystories_vocab_size,
        )
        cfg.vocab_size = _train_data.vocab_size
        print(
            f"Data: tinystories-bpe | repo={tinystories_repo} "
            f"| vocab={_train_data.vocab_size} | train={len(_train_data):,} "
            f"| val={len(_val_data):,} tokens"
        )
        return

    path = os.path.join(data_dir, "tiny_shakespeare.txt")
    if os.path.exists(path):
        _train_data, _val_data = load_shakespeare(
            seq_len=cfg.max_seq_len, device=device, data_dir=data_dir
        )
        print(f"Data: tiny_shakespeare | vocab={_train_data.vocab_size} "
              f"| train={len(_train_data):,} | val={len(_val_data):,} tokens")
        # Override vocab_size in config to match actual data
        cfg.vocab_size = _train_data.vocab_size
    else:
        print("Data: random (tiny_shakespeare.txt not found in data_cache/)")


def get_batch(cfg: ModelConfig, device: str = "cpu", split: str = "train"):
    """Get a batch — real Shakespeare data if loaded, else random."""
    if split == "val" and _val_data is not None:
        return _val_data.get_batch(cfg.batch_size)
    if _train_data is not None:
        return _train_data.get_batch(cfg.batch_size)
    # Fallback: random
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=device)
    return x, y


def _build_optimizer(model: nn.Module, cfg: ModelConfig) -> torch.optim.AdamW:
    """Build optimizer with separate param group for router (lower LR)."""
    router_param_ids = set()
    router_params = []

    # Collect router / equilibrium parameters
    for name, p in model.named_parameters():
        if any(k in name for k in ("eq_signals", "router")):
            router_param_ids.add(id(p))
            router_params.append(p)

    backbone_params = [p for p in model.parameters() if id(p) not in router_param_ids]

    if router_params:
        param_groups = [
            {"params": backbone_params, "lr": cfg.learning_rate},
            {"params": router_params, "lr": cfg.router_lr},
        ]
    else:
        param_groups = [{"params": backbone_params, "lr": cfg.learning_rate}]

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def _format_param_count(model: nn.Module) -> str:
    n_params = sum(p.numel() for p in model.parameters())
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.1f}M"
    return f"{n_params / 1_000:.1f}K"


def build_model(cfg: ModelConfig, stage: str, device: str) -> nn.Module:
    if stage == "b":
        return ABPTModelB(cfg).to(device)
    if stage == "anchor":
        return ABPTAnchorV1(cfg).to(device)
    return ABPTModel(cfg).to(device)


def _summarize_step_metrics(
    out: dict,
    loss_value: float,
    stage: str,
    val_bpb: float | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "loss": float(loss_value),
    }
    if "ce_loss" in out:
        metrics["ce_loss"] = float(out["ce_loss"].item())
        metrics["bpb"] = float(bits_per_byte(out["ce_loss"].item()))
    if val_bpb is not None:
        metrics["val_bpb"] = float(val_bpb)
    if "diversity_loss" in out:
        metrics["diversity_loss"] = float(out["diversity_loss"].item())
    if "branch_logits" in out:
        metrics["branch_diversity"] = float(branch_diversity(out["branch_logits"]))
    if "confidence" in out:
        metrics["confidence"] = float(out["confidence"].mean().item())
    if stage == "b" and "route_stats" in out and out["route_stats"]:
        rs = out["route_stats"][-1]
        metrics["mean_ed"] = float(rs["mean_ed"])
        metrics["route_forward"] = float(rs["forward"])
        metrics["route_backward"] = float(rs["backward"])
    if "anchor_diagnostics" in out:
        diag = out["anchor_diagnostics"]
        metrics["anchors_active"] = float(diag["num_active"])
        metrics["anchor_contradiction"] = float(diag["mean_contradiction_pressure"])
        metrics["anchor_viability"] = float(diag["mean_viability"])
        metrics["anchor_dead_end"] = float(diag["dead_end_count"])
    if "proposal_diagnostics" in out:
        diag = out["proposal_diagnostics"]
        metrics["proposal_influence"] = float(diag["anchors_with_proposal_influence"])
        metrics["proposal_blend"] = float(diag["mean_blend_ratio"])
        metrics["strong_retire_gap"] = float(diag["mean_strong_retire_gap"])
    if "component_losses" in out:
        component_losses = out["component_losses"]
        if "detector_alignment_loss" in component_losses:
            metrics["detector_alignment_loss"] = float(component_losses["detector_alignment_loss"].item())
        if "context_stability_loss" in component_losses:
            metrics["context_stability_loss"] = float(component_losses["context_stability_loss"].item())
    return metrics


def _save_history(history: list[dict[str, float]], history_path: str) -> None:
    directory = os.path.dirname(history_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _router_entropy_loss(route_probs: torch.Tensor) -> torch.Tensor:
    """Auxiliary loss penalizing route distribution too far from target entropy.

    Encourages specialization (not uniform) while still using all routes.
    target = log(2) ≈ 0.693 — ideal 50/50 split between two dominant routes.
    """
    target_entropy = math.log(2.0)
    # route_probs: [B, T, 4] — mean over batch and sequence
    mean_probs = route_probs.mean(dim=(0, 1))  # [4]
    actual_entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum()
    return (actual_entropy - target_entropy).abs()


def train(
    cfg: ModelConfig,
    device: str = "cpu",
    stage: str = "a",
    data_dir: str = "data_cache",
    dataset: str = "shakespeare",
    history_path: str | None = None,
    the_stack_repo: str = "bigcode/the-stack-smol-xs",
    the_stack_lang: str = "python",
    the_stack_bytes: int = 8_000_000,
    the_stack_vocab_size: int = 4096,
    tinystories_repo: str = "roneneldan/TinyStories",
    tinystories_bytes: int = 16_000_000,
    tinystories_vocab_size: int = 4096,
):
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
    history: list[dict[str, float]] = []

    optimizer = _build_optimizer(model, cfg)

    print(f"Model: {_format_param_count(model)} params (Stage {stage.upper()})")
    print(f"Config: attnres={cfg.use_attn_res} branches={cfg.use_branches} "
          f"verifier={cfg.use_verifier} plastic={cfg.use_plastic}")
    if stage == "b":
        print(f"Router: lr={cfg.router_lr} warmup={cfg.router_warmup_steps} "
              f"eq_momentum={cfg.eq_momentum} eq_warmup={cfg.eq_warmup_steps}")
    print(f"Device: {device}")
    print("---")

    for step in range(cfg.max_steps):
        model.train()
        x, y = get_batch(cfg, device, split="train")

        out = model(x, y)
        loss = out["loss"]

        # Phase 0.4: router entropy auxiliary loss (Stage B only)
        if stage == "b" and "route_stats" in out:
            # Get route_probs from the last forward — need to recompute from model
            # We add entropy loss on the diversity_loss channel
            last_eq = model.eq_signals[-1]
            if not last_eq.is_warming_up:
                # Re-derive route_probs from last layer's ED
                with torch.no_grad():
                    eq_out = last_eq(out.get("_last_hidden", model.ln_final(x)))
                route_out = model.router(eq_out["ed"])
                ent_loss = _router_entropy_loss(route_out["route_probs"])
                loss = loss + cfg.router_entropy_weight * ent_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()

        if cfg.use_plastic and hasattr(model, "plastic"):
            model.plastic.apply_decay()

        if step % cfg.eval_interval == 0:
            # Train metrics
            bpb = bits_per_byte(out["ce_loss"].item())
            msg = f"step {step:5d} | loss {loss.item():.4f} | bpb {bpb:.4f}"
            val_bpb: float | None = None

            # Val bpb on real data
            if _val_data is not None:
                model.eval()
                with torch.no_grad():
                    xv, yv = get_batch(cfg, device, split="val")
                    val_out = model(xv, yv)
                    val_bpb = bits_per_byte(val_out["ce_loss"].item())
                msg += f" | val_bpb {val_bpb:.4f}"
                model.train()

            if "diversity_loss" in out:
                msg += f" | div {out['diversity_loss'].item():.4f}"
            if "branch_logits" in out:
                bd = branch_diversity(out["branch_logits"])
                msg += f" | bdiv {bd:.4f}"
            if "confidence" in out:
                msg += f" | conf {out['confidence'].mean().item():.4f}"
            if "route_stats" in out and out["route_stats"]:
                rs = out["route_stats"][-1]
                msg += f" | ed={rs['mean_ed']:.3f} fwd={rs['forward']:.2f} bk={rs['backward']:.2f}"
            if "anchor_diagnostics" in out:
                anchor_diag = out["anchor_diagnostics"]
                msg += (
                    f" | anchors {anchor_diag['num_active']}"
                    f" | cpress {anchor_diag['mean_contradiction_pressure']:.3f}"
                    f" | viable {anchor_diag['mean_viability']:.3f}"
                    f" | dead {anchor_diag['dead_end_count']}"
                )
            if "proposal_diagnostics" in out:
                proposal_diag = out["proposal_diagnostics"]
                msg += (
                    f" | prop {proposal_diag['anchors_with_proposal_influence']}"
                    f" | blend {proposal_diag['mean_blend_ratio']:.3f}"
                    f" | srgap {proposal_diag['mean_strong_retire_gap']:.3f}"
                )
            if "component_losses" in out:
                component_losses = out["component_losses"]
                if "detector_alignment_loss" in component_losses:
                    msg += f" | dalign {component_losses['detector_alignment_loss'].item():.4f}"
                if "context_stability_loss" in component_losses:
                    msg += f" | cstab {component_losses['context_stability_loss'].item():.4f}"
            print(msg)
            step_metrics = _summarize_step_metrics(
                out=out,
                loss_value=loss.item(),
                stage=stage,
                val_bpb=val_bpb,
            )
            step_metrics["step"] = float(step)
            history.append(step_metrics)

    print("---")
    print(f"Final loss: {loss.item():.4f}")
    model.training_history = history
    if history_path is not None:
        _save_history(history, history_path)
    return model


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, device: str = "cpu") -> str:
    """Sample text from model given a prompt string."""
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            ids_crop = ids[:, -model.cfg.max_seq_len:]
            out = model(ids_crop)
            logits = out["logits"][:, -1, :]  # [1, V]
            probs = torch.softmax(logits / 0.8, dim=-1)  # temperature=0.8
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
    return tokenizer.decode(ids[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABPT Training")
    parser.add_argument("--preset", default="toy", choices=list(PRESETS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
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
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--history_path", default=None)
    parser.add_argument("--sample", action="store_true", help="Generate sample text after training")
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len
    model = train(
        cfg,
        args.device,
        args.stage,
        args.data_dir,
        args.dataset,
        args.history_path,
        args.the_stack_repo,
        args.the_stack_lang,
        args.the_stack_bytes,
        args.the_stack_vocab_size,
        args.tinystories_repo,
        args.tinystories_bytes,
        args.tinystories_vocab_size,
    )

    if args.sample and _train_data is not None:
        print("\n--- Sample output ---")
        prompt = "\nFIRST CITIZEN:\n"
        sample = generate(model, _train_data.tokenizer, prompt, max_new_tokens=300, device=args.device)
        print(sample)
