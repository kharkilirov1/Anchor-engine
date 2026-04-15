from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_motif_synthetic import DOMAIN_NAMES, QwenMotifSyntheticDataset
from src.model.qwen_motif_config import (
    QwenMotifFullConfig,
    QwenMotifPatchConfig,
    QwenMotifRouterConfig,
    build_default_attention_patch_config,
    build_default_ffn_lora_configs,
)
from src.model.qwen_motif_patch import (
    apply_qwen_motif_pipeline,
    build_and_patch_qwen_attention_layers,
    build_and_patch_qwen_ffn_layers,
    collect_qwen_motif_attention_adapters,
    collect_qwen_motif_mlps,
    collect_qwen_motif_trainable_names,
    freeze_model_except_motif_routers,
    freeze_model_except_qwen_motif_trainables,
    partial_reinit_qwen_motif_modules,
)


def build_tiny_qwen_model(seq_len: int, vocab_size: int) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=max(seq_len, 64),
    )
    model = Qwen2ForCausalLM(config)
    model.float()
    return model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def attention_router_collapsed(eval_stats: dict[str, Any], threshold: float = 1.9) -> bool:
    values: list[float] = []
    for domain_stats in eval_stats.values():
        for layer_values in domain_stats.get("attn_mean_alpha", {}).values():
            values.extend(float(value) for value in layer_values)
    if not values:
        return False
    return max(values) >= threshold


def evaluate_domain_stats(model: Qwen2ForCausalLM, dataset: QwenMotifSyntheticDataset, batch_size: int) -> dict[str, Any]:
    model.eval()
    stats: dict[str, Any] = {}
    with torch.no_grad():
        for domain in DOMAIN_NAMES:
            input_ids, _ = dataset.get_batch(batch_size=batch_size, domain=domain)
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False, return_dict=True)
            domain_stats: dict[str, Any] = {"loss": float(outputs.loss.item())}
            ffn_stats = {}
            for layer_id, module in collect_qwen_motif_mlps(model).items():
                router_stats = module.get_last_router_stats()
                if router_stats:
                    ffn_stats[str(layer_id)] = [float(value) for value in router_stats["mean_alpha"].tolist()]
            attn_stats = {}
            for layer_id, module in collect_qwen_motif_attention_adapters(model).items():
                router_stats = module.get_last_router_stats()
                if router_stats:
                    attn_stats[str(layer_id)] = [float(value) for value in router_stats["mean_alpha"].tolist()]
            domain_stats["ffn_mean_alpha"] = ffn_stats
            domain_stats["attn_mean_alpha"] = attn_stats
            stats[domain] = domain_stats
    return stats


def train_stage(
    model: Qwen2ForCausalLM,
    dataset: QwenMotifSyntheticDataset,
    steps: int,
    batch_size: int,
    lr: float,
    stage_name: str,
) -> list[dict[str, Any]]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        return []
    optimizer = AdamW(params, lr=lr)
    history: list[dict[str, Any]] = []
    model.train()
    for step in range(steps):
        input_ids, domains = dataset.get_batch(batch_size=batch_size)
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False, return_dict=True)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        history.append({"stage": stage_name, "step": step, "loss": float(loss.item()), "domains": domains})
    return history


def run_pretrained_smoke(model_name: str, device: str) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    patch_config = QwenMotifPatchConfig(
        layer_ids=(0, 1),
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=128),
        freeze_base=True,
        freeze_model=True,
    )
    build_and_patch_qwen_ffn_layers(model, patch_config)
    build_and_patch_qwen_attention_layers(model, build_default_attention_patch_config(layer_ids=(0, 1), rank=4, alpha=8.0))
    freeze_model_except_qwen_motif_trainables(model)
    prompt = "Write a short Python function that returns the maximum of two integers."
    batch = tokenizer(prompt, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, use_cache=False, return_dict=True)
    return {
        "model": model_name,
        "device": device,
        "prompt_length": int(batch["input_ids"].shape[1]),
        "logits_shape": list(outputs.logits.shape),
        "trainable_parameters": count_trainable_parameters(model),
        "ffn_layers": sorted(collect_qwen_motif_mlps(model).keys()),
        "attention_layers": sorted(collect_qwen_motif_attention_adapters(model).keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged Qwen motif smoke experiments.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--router_steps", type=int, default=12)
    parser.add_argument("--lora_steps", type=int, default=12)
    parser.add_argument("--attention_steps", type=int, default=12)
    parser.add_argument("--sparse_steps", type=int, default=8)
    parser.add_argument("--reinit_fraction", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run_pretrained_smoke", action="store_true")
    parser.add_argument("--pretrained_model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output_json", type=Path, default=ROOT / "archive" / "qwen_motif_stage_experiments.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = QwenMotifSyntheticDataset(seq_len=args.seq_len, vocab_size=args.vocab_size, seed=args.seed, device=args.device)
    base_model = build_tiny_qwen_model(seq_len=args.seq_len, vocab_size=args.vocab_size).to(args.device)

    router_model = deepcopy(base_model)
    router_config = QwenMotifPatchConfig(
        layer_ids=(1, 2),
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=48),
        freeze_base=True,
        freeze_model=True,
    )
    build_and_patch_qwen_ffn_layers(router_model, router_config)
    freeze_model_except_motif_routers(router_model)
    router_history = train_stage(router_model, dataset, args.router_steps, args.batch_size, args.lr, "router_only")
    router_eval = evaluate_domain_stats(router_model, dataset, args.batch_size)

    lora_model = deepcopy(base_model)
    lora_config = QwenMotifPatchConfig(
        layer_ids=(1, 2),
        router=QwenMotifRouterConfig(router_type="contextual", hidden_size=48),
        freeze_base=True,
        freeze_model=True,
        expert_lora=build_default_ffn_lora_configs(rank=4, alpha=8.0),
    )
    build_and_patch_qwen_ffn_layers(lora_model, lora_config)
    freeze_model_except_qwen_motif_trainables(lora_model)
    lora_history = train_stage(lora_model, dataset, args.lora_steps, args.batch_size, args.lr, "ffn_lora")
    lora_eval = evaluate_domain_stats(lora_model, dataset, args.batch_size)

    full_model = deepcopy(base_model)
    apply_qwen_motif_pipeline(
        full_model,
        QwenMotifFullConfig(
            ffn=lora_config,
            attention=build_default_attention_patch_config(layer_ids=(1, 2), rank=4, alpha=8.0, top_k=None),
        ),
    )
    freeze_model_except_qwen_motif_trainables(full_model)
    attention_history = train_stage(full_model, dataset, args.attention_steps, args.batch_size, args.lr, "attention_soft")
    attention_eval = evaluate_domain_stats(full_model, dataset, args.batch_size)

    sparse_model = deepcopy(full_model)
    for adapter in collect_qwen_motif_attention_adapters(sparse_model).values():
        adapter.router.top_k = 1
    sparse_reinit_applied = attention_router_collapsed(attention_eval)
    if sparse_reinit_applied:
        partial_reinit_qwen_motif_modules(sparse_model, fraction=args.reinit_fraction)
    freeze_model_except_qwen_motif_trainables(sparse_model)
    sparse_history = train_stage(sparse_model, dataset, args.sparse_steps, args.batch_size, args.lr, "attention_sparse")
    sparse_eval = evaluate_domain_stats(sparse_model, dataset, args.batch_size)

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "seed": args.seed,
        "resource_note": "cpu-only smoke run" if args.device == "cpu" else "accelerated run",
        "router_only": {
            "trainable_parameters": count_trainable_parameters(router_model),
            "trainable_names": collect_qwen_motif_trainable_names(router_model),
            "history": router_history,
            "eval": router_eval,
        },
        "ffn_lora": {
            "trainable_parameters": count_trainable_parameters(lora_model),
            "trainable_names": collect_qwen_motif_trainable_names(lora_model),
            "history": lora_history,
            "eval": lora_eval,
        },
        "attention_soft": {
            "trainable_parameters": count_trainable_parameters(full_model),
            "trainable_names": collect_qwen_motif_trainable_names(full_model),
            "history": attention_history,
            "eval": attention_eval,
        },
        "attention_sparse": {
            "trainable_parameters": count_trainable_parameters(sparse_model),
            "trainable_names": collect_qwen_motif_trainable_names(sparse_model),
            "history": sparse_history,
            "eval": sparse_eval,
            "partial_reinit_applied": sparse_reinit_applied,
            "reinit_fraction": args.reinit_fraction,
        },
    }
    if args.run_pretrained_smoke:
        payload["pretrained_smoke"] = run_pretrained_smoke(args.pretrained_model, args.device)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_json={args.output_json}")
    print(json.dumps({
        "router_only_last_loss": router_history[-1]["loss"] if router_history else None,
        "ffn_lora_last_loss": lora_history[-1]["loss"] if lora_history else None,
        "attention_soft_last_loss": attention_history[-1]["loss"] if attention_history else None,
        "attention_sparse_last_loss": sparse_history[-1]["loss"] if sparse_history else None,
        "pretrained_smoke": payload.get("pretrained_smoke"),
    }, indent=2))


if __name__ == "__main__":
    main()
