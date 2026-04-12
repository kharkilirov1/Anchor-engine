from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_module
from scripts.run_abpt_autoresearch_loop import summarize_history
from src.model.config import PRESETS, ModelConfig


@dataclass(frozen=True)
class DomainSpec:
    key: str
    dataset: str
    seq_len: int = 32
    batch_size: int = 4
    steps: int = 100
    eval_interval: int = 10
    tinystories_bytes: int = 180_000
    tinystories_vocab_size: int = 256
    the_stack_repo: str = "bigcode/the-stack-smol-xs"
    the_stack_lang: str = "python"
    the_stack_bytes: int = 200_000
    the_stack_vocab_size: int = 256
    openwebmath_repo: str = "open-web-math/open-web-math"
    openwebmath_bytes: int = 200_000
    openwebmath_vocab_size: int = 256


@dataclass(frozen=True)
class ModelSpec:
    key: str
    preset: str
    stage: str
    use_fog_flow: bool = False
    use_proposal_rollout: bool = True


@dataclass(frozen=True)
class CapacityMatch:
    cfg: ModelConfig
    param_count: int
    target_params: int

    @property
    def param_delta(self) -> int:
        return self.param_count - self.target_params


def build_domain_specs(steps: int) -> list[DomainSpec]:
    return [
        DomainSpec(key="stories", dataset="tinystories-bpe", steps=steps),
        DomainSpec(key="code", dataset="the-stack-bpe", steps=steps),
        DomainSpec(key="math", dataset="openwebmath-bpe", steps=steps),
    ]


def build_model_specs(rollout_models: set[str] | None = None) -> list[ModelSpec]:
    rollout_models = rollout_models or {"anchor", "anchor_fog"}
    return [
        ModelSpec(
            key="baseline",
            preset="baseline-0",
            stage="a",
            use_fog_flow=False,
            use_proposal_rollout=False,
        ),
        ModelSpec(
            key="anchor",
            preset="full",
            stage="anchor",
            use_fog_flow=False,
            use_proposal_rollout="anchor" in rollout_models,
        ),
        ModelSpec(
            key="anchor_fog",
            preset="full",
            stage="anchor",
            use_fog_flow=True,
            use_proposal_rollout="anchor_fog" in rollout_models,
        ),
    ]


def _expected_vocab_size(domain_spec: DomainSpec) -> int:
    if domain_spec.dataset == "tinystories-bpe":
        return domain_spec.tinystories_vocab_size
    if domain_spec.dataset == "the-stack-bpe":
        return domain_spec.the_stack_vocab_size
    if domain_spec.dataset == "openwebmath-bpe":
        return domain_spec.openwebmath_vocab_size
    raise ValueError(f"Unsupported dataset for real-domain sweep: {domain_spec.dataset}")


def _resolve_anchor_domain_mode(dataset: str) -> str:
    return "synthetic" if dataset == "anchor-synthetic" else "real"


def _resolve_fog_task_profile(dataset: str) -> str:
    if dataset in {"the-stack", "the-stack-bpe"}:
        return "code"
    if dataset == "openwebmath-bpe":
        return "math"
    if dataset == "tinystories-bpe":
        return "stories"
    if dataset == "anchor-synthetic":
        return "synthetic"
    return "balanced"


def _base_cfg_for_model(model_spec: ModelSpec, domain_spec: DomainSpec) -> ModelConfig:
    return replace(
        PRESETS[model_spec.preset],
        batch_size=domain_spec.batch_size,
        max_steps=domain_spec.steps,
        eval_interval=domain_spec.eval_interval,
        max_seq_len=domain_spec.seq_len,
        vocab_size=_expected_vocab_size(domain_spec),
        anchor_domain_mode=_resolve_anchor_domain_mode(domain_spec.dataset),
        fog_task_profile=_resolve_fog_task_profile(domain_spec.dataset),
        use_fog_flow=model_spec.use_fog_flow,
        anchor_use_proposal_rollout=model_spec.use_proposal_rollout,
    )


def _count_model_params(model_spec: ModelSpec, cfg: ModelConfig) -> int:
    model = train_module.build_model(cfg, model_spec.stage, "cpu")
    return sum(p.numel() for p in model.parameters())


def _search_d_ff_values(cfg: ModelConfig, use_fog_flow: bool) -> list[int]:
    if use_fog_flow:
        return [cfg.d_ff]
    center = max(cfg.d_model, int(round((cfg.d_ff / 32))) * 32)
    values = {cfg.d_ff, center}
    for delta in (-64, -32, 0, 32, 64):
        values.add(max(cfg.d_model, center + delta))
    return sorted(values)


def match_model_capacity(
    model_spec: ModelSpec,
    domain_spec: DomainSpec,
    target_params: int,
) -> CapacityMatch:
    base_cfg = _base_cfg_for_model(model_spec, domain_spec)
    default_count = _count_model_params(model_spec, base_cfg)

    best_cfg = base_cfg
    best_count = default_count
    best_score = (
        abs(default_count - target_params),
        abs(base_cfg.d_model - PRESETS[model_spec.preset].d_model),
        abs(base_cfg.d_ff - PRESETS[model_spec.preset].d_ff),
    )
    scaled_d_model = int(round(base_cfg.d_model * (target_params / max(default_count, 1)) ** 0.5))
    center_d_model = max(base_cfg.n_heads * 8, (scaled_d_model // base_cfg.n_heads) * base_cfg.n_heads)
    if center_d_model < scaled_d_model:
        center_d_model += base_cfg.n_heads
    search_values = {base_cfg.d_model, center_d_model}
    for delta in range(-48, 49, base_cfg.n_heads):
        search_values.add(max(base_cfg.n_heads * 8, center_d_model + delta))
    candidate_d_models = sorted(value for value in search_values if 96 <= value <= 512)

    for d_model in candidate_d_models:
        candidate_cfg = replace(
            base_cfg,
            d_model=d_model,
            d_ff=max(d_model, int(round(base_cfg.d_ff * (d_model / max(base_cfg.d_model, 1)) / 32)) * 32),
            plastic_hidden=max(16, d_model // 4),
        )
        for d_ff in _search_d_ff_values(candidate_cfg, model_spec.use_fog_flow):
            search_cfg = replace(candidate_cfg, d_ff=max(d_model, d_ff))
            param_count = _count_model_params(model_spec, search_cfg)
            score = (
                abs(param_count - target_params),
                abs(d_model - PRESETS[model_spec.preset].d_model),
                abs(search_cfg.d_ff - PRESETS[model_spec.preset].d_ff),
            )
            if score < best_score:
                best_cfg = search_cfg
                best_count = param_count
                best_score = score

    return CapacityMatch(cfg=best_cfg, param_count=best_count, target_params=target_params)


def build_capacity_matches(
    domain_spec: DomainSpec,
    target_params: int | None = None,
    rollout_models: set[str] | None = None,
) -> tuple[dict[str, CapacityMatch], dict[str, int], int]:
    model_specs = build_model_specs(rollout_models)
    default_counts = {
        model_spec.key: _count_model_params(model_spec, _base_cfg_for_model(model_spec, domain_spec))
        for model_spec in model_specs
    }
    if target_params is None:
        target_params = round(fmean(default_counts.values()))
    matches = {
        model_spec.key: match_model_capacity(model_spec, domain_spec, target_params)
        for model_spec in model_specs
    }
    return matches, default_counts, target_params


def run_once(
    model_spec: ModelSpec,
    domain_spec: DomainSpec,
    output_dir: Path,
    capacity_match: CapacityMatch,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    cfg = replace(
        capacity_match.cfg,
        batch_size=domain_spec.batch_size,
        max_steps=domain_spec.steps,
        eval_interval=domain_spec.eval_interval,
        max_seq_len=domain_spec.seq_len,
    )
    history_path = output_dir / f"{model_spec.key}_{domain_spec.key}.history.json"
    model = train_module.train(
        cfg=cfg,
        device="cpu",
        stage=model_spec.stage,
        data_dir=str(ROOT / "data_cache"),
        dataset=domain_spec.dataset,
        history_path=str(history_path),
        tinystories_bytes=domain_spec.tinystories_bytes,
        tinystories_vocab_size=domain_spec.tinystories_vocab_size,
        the_stack_repo=domain_spec.the_stack_repo,
        the_stack_lang=domain_spec.the_stack_lang,
        the_stack_bytes=domain_spec.the_stack_bytes,
        the_stack_vocab_size=domain_spec.the_stack_vocab_size,
        openwebmath_repo=domain_spec.openwebmath_repo,
        openwebmath_bytes=domain_spec.openwebmath_bytes,
        openwebmath_vocab_size=domain_spec.openwebmath_vocab_size,
    )
    result = {
        "model": asdict(model_spec),
        "domain": asdict(domain_spec),
        "param_count": sum(p.numel() for p in model.parameters()),
        "seed": seed,
        "matched_cfg": {
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "plastic_hidden": cfg.plastic_hidden,
            "anchor_use_proposal_rollout": cfg.anchor_use_proposal_rollout,
            "target_params": capacity_match.target_params,
            "param_delta": sum(p.numel() for p in model.parameters()) - capacity_match.target_params,
        },
        "summary": summarize_history(model.training_history),
        "history_path": str(history_path),
    }
    result_path = output_dir / f"{model_spec.key}_{domain_spec.key}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def write_summary(
    output_dir: Path,
    results: list[dict[str, Any]],
    capacity_metadata: dict[str, dict[str, Any]],
) -> None:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    by_model: dict[str, list[float]] = {}
    for result in results:
        domain_key = result["domain"]["key"]
        model_key = result["model"]["key"]
        by_domain.setdefault(domain_key, []).append(result)
        by_model.setdefault(model_key, []).append(result["summary"]["best_val_bpb"])

    lines = ["# Anchor + FOG real-domain sweep (equal params)", ""]
    lines.append("## Matched model configs by domain")
    for domain_key, meta in sorted(capacity_metadata.items()):
        lines.append("")
        lines.append(f"### {domain_key} target params: **{meta['target_params']:,}**")
        lines.append("")
        lines.append("| Model | Default Params | Matched Params | Delta | d_model | d_ff | Rollout |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for model_key, match in meta["matches"].items():
            lines.append(
                f"| {model_key} | {meta['default_counts'][model_key]:,} | {match.param_count:,} | "
                f"{match.param_delta:+,} | {match.cfg.d_model} | {match.cfg.d_ff} | "
                f"{match.cfg.anchor_use_proposal_rollout} |"
            )

    lines.append("")
    lines.append("## Overall mean best val BPB")
    lines.append("")
    lines.append("| Model | Mean Best Val BPB | Runs |")
    lines.append("|---|---:|---:|")
    for model_key, scores in sorted(by_model.items(), key=lambda item: fmean(item[1])):
        lines.append(f"| {model_key} | {fmean(scores):.4f} | {len(scores)} |")

    for domain_key, rows in sorted(by_domain.items()):
        lines.append("")
        lines.append(f"## {domain_key}")
        lines.append("")
        lines.append("| Model | Stage | FOG flow | Params | Best Val BPB | Final Val BPB |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in sorted(rows, key=lambda item: item["summary"]["best_val_bpb"]):
            lines.append(
                f"| {row['model']['key']} | {row['model']['stage']} | "
                f"{row['model']['use_fog_flow']} | {row['param_count']:,} | "
                f"{row['summary']['best_val_bpb']:.4f} | {row['summary']['final_val_bpb']:.4f} |"
            )

    summary_md = output_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "capacity_metadata": {
                    domain_key: {
                        "target_params": meta["target_params"],
                        "default_counts": meta["default_counts"],
                        "matches": {
                            key: {
                                "param_count": match.param_count,
                                "param_delta": match.param_delta,
                                "cfg": {
                                    "d_model": match.cfg.d_model,
                                    "d_ff": match.cfg.d_ff,
                                    "plastic_hidden": match.cfg.plastic_hidden,
                                    "anchor_use_proposal_rollout": match.cfg.anchor_use_proposal_rollout,
                                },
                            }
                            for key, match in meta["matches"].items()
                        },
                    }
                    for domain_key, meta in capacity_metadata.items()
                },
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs anchor vs anchor+fog on real domains.")
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "anchor_fog_real_domains_v2_equal_params"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--target_params", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--rollout-models",
        nargs="*",
        default=["anchor", "anchor_fog"],
        choices=["anchor", "anchor_fog"],
        help="Model keys that instantiate the proposal rollout branch.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_specs = build_domain_specs(args.steps)
    rollout_models = set(args.rollout_models)
    results = []
    capacity_metadata: dict[str, dict[str, Any]] = {}
    for domain_idx, domain_spec in enumerate(domain_specs):
        capacity_matches, default_counts, target_params = build_capacity_matches(
            domain_spec=domain_spec,
            target_params=args.target_params,
            rollout_models=rollout_models,
        )
        capacity_metadata[domain_spec.key] = {
            "matches": capacity_matches,
            "default_counts": default_counts,
            "target_params": target_params,
        }
        for model_idx, model_spec in enumerate(build_model_specs(rollout_models)):
            print(f"\n=== {domain_spec.key} :: {model_spec.key} ===")
            results.append(
                run_once(
                    model_spec=model_spec,
                    domain_spec=domain_spec,
                    output_dir=output_dir,
                    capacity_match=capacity_matches[model_spec.key],
                    seed=args.seed + domain_idx * 100 + model_idx,
                )
            )

    write_summary(output_dir, results, capacity_metadata)
    print(json.dumps({"output_dir": str(output_dir), "runs": len(results)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
