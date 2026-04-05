"""
ABPT Expanded Diagnostic — расширенный набор кейсов
====================================================
Прогоняет все 29 кейсов (13 старых + 16 новых) через geometry pipeline:
  1. Извлекает r1 профиль по всем слоям
  2. Классифицирует в 3 кластера (mature / template / flat)
  3. Генерирует base + anchor текст
  4. Считает constraint_score
  5. Считает rescue rate по кластерам
  6. Считает корреляции tail_retention_ratio vs constraint

Вывод:
  archive/qwen35_4b_expanded_diagnostic_<profile>.json

Использование:
  python scripts/run_qwen_expanded_diagnostic.py --anchor-profile medium
  python scripts/run_qwen_expanded_diagnostic.py --anchor-profile short --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import (
    list_anchor_span_profiles,
    make_qwen_anchor_geometry_cases,
    QwenAnchorGeometryCase,
)
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import (
    build_tail_reference_layers,
    compute_geometry_metrics,
    extract_delta_vectors,
    match_anchor_span,
    select_tail_probe_layers,
)

# Import scoring and phase metrics from existing phase probe
from scripts.run_qwen_phase_probe import (
    KEYWORD_SPECS,
    compute_phase_metrics,
    score_constraint,
    spearman_correlation,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_LENGTH = 160
MAX_NEW_TOKENS = 120
SEED = 7

DEFAULT_MATURE_R1_THRESHOLD = 0.65
DEFAULT_TEMPLATE_DELTA_THRESHOLD = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Cluster classification (same logic as qwen_anchor_overlay.py)
# ─────────────────────────────────────────────────────────────────────────────

def classify_cluster(
    r1_profile: dict[str, float | None],
    reference_layers: dict[str, int],
    mature_threshold: float = DEFAULT_MATURE_R1_THRESHOLD,
    template_threshold: float = DEFAULT_TEMPLATE_DELTA_THRESHOLD,
) -> dict[str, Any]:
    mature_layer = reference_layers["mature_layer"]
    template_prev = reference_layers["template_prev_layer"]
    template_curr = reference_layers["template_curr_layer"]

    r1_ref = r1_profile.get(str(mature_layer))
    r1_prev = r1_profile.get(str(template_prev))
    r1_curr = r1_profile.get(str(template_curr))

    delta_template = None
    if r1_prev is not None and r1_curr is not None:
        delta_template = float(r1_curr) - float(r1_prev)

    if r1_ref is not None and float(r1_ref) > mature_threshold:
        cluster = "mature"
        route = "guided_lite"
    elif delta_template is not None and delta_template > template_threshold:
        cluster = "template"
        route = "trust"
    else:
        cluster = "flat"
        route = "anchor_forced"

    return {
        "cluster": cluster,
        "route": route,
        "r1_reference": float(r1_ref) if r1_ref is not None else None,
        "delta_template_pair": delta_template,
        "mature_layer": mature_layer,
        "template_prev_layer": template_prev,
        "template_curr_layer": template_curr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_full_r1_profile(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    probe_layers: list[int],
    device: torch.device,
) -> dict[str, float | None] | None:
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    offsets = None
    try:
        encoded = tokenizer(
            case.prompt,
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [
            (int(s), int(e))
            for s, e in encoded.pop("offset_mapping")[0].tolist()
        ]
    except TypeError:
        encoded = tokenizer(
            case.prompt,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    batch = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    input_ids = [int(t) for t in batch["input_ids"][0].tolist()]

    span_match = match_anchor_span(
        text=case.prompt,
        anchor_text=case.anchor_text,
        input_ids=input_ids,
        tokenizer=tokenizer,
        offsets=offsets,
    )
    if span_match is None:
        return None

    with torch.no_grad():
        outputs = overlay.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states

    r1_profile: dict[str, float | None] = {}
    for layer in probe_layers:
        if layer + 1 >= len(hidden_states):
            r1_profile[str(layer)] = None
            continue
        delta_vecs = extract_delta_vectors(
            hidden_states[layer + 1][0],
            span_match.token_start,
            span_match.token_end,
        )
        metrics = compute_geometry_metrics(delta_vecs)
        val = metrics.get("rank1_explained_variance")
        r1_profile[str(layer)] = float(val) if val is not None else None

    return r1_profile


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(
    overlay: QwenAnchorOverlay,
    prompt: str,
    use_anchor: bool = False,
) -> str:
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    device = next(overlay.parameters()).device
    encoded = tokenizer(
        [prompt],
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    n_prompt = int(input_ids.shape[1])

    if use_anchor and hasattr(overlay, "generate_with_anchor_bias"):
        generated = overlay.generate_with_anchor_bias(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    else:
        with torch.no_grad():
            generated = overlay.base_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
    continuation = generated[0][n_prompt:]
    return tokenizer.decode(continuation, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    model_name: str,
    anchor_profile: str,
    device_str: str,
    mature_threshold: float,
    template_threshold: float,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    device = torch.device(device_str)

    print(f"[ExpandedDiag] Model:   {model_name}")
    print(f"[ExpandedDiag] Profile: {anchor_profile}")
    print(f"[ExpandedDiag] Device:  {device}")

    print("[ExpandedDiag] Loading model...")
    overlay = QwenAnchorOverlay.from_pretrained(model_name, config=TOY_CONFIG)
    overlay.to(device)
    overlay.eval()

    n_layers = int(overlay.model_num_hidden_layers)
    probe_layers = list(range(n_layers))
    reference_layers = build_tail_reference_layers(
        select_tail_probe_layers(n_layers, count=10)
    )

    cases = make_qwen_anchor_geometry_cases(anchor_span_profile=anchor_profile)
    print(f"[ExpandedDiag] Cases: {len(cases)}")
    print(f"[ExpandedDiag] Reference layers: {reference_layers}")

    results: list[dict[str, Any]] = []
    cluster_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case.name} ({case.anchor_group})")

        # --- r1 profile ---
        r1_profile = extract_full_r1_profile(overlay, case, probe_layers, device)
        if r1_profile is None:
            print("  SKIP (span not found)")
            continue

        # --- cluster classification ---
        cluster_info = classify_cluster(
            r1_profile, reference_layers,
            mature_threshold=mature_threshold,
            template_threshold=template_threshold,
        )
        cluster = cluster_info["cluster"]

        # --- phase metrics ---
        phase_metrics = compute_phase_metrics(r1_profile, n_layers)

        # --- base generation ---
        print(f"  cluster={cluster} | generating base...")
        base_text = generate_text(overlay, case.prompt, use_anchor=False)
        base_constraint = score_constraint(base_text, case.anchor_group)
        base_score = base_constraint["constraint_score"]

        # --- anchor generation (for flat cluster) ---
        anchor_text_gen = ""
        anchor_score = 0.0
        if cluster == "flat":
            print("  cluster=flat → generating anchor...")
            anchor_text_gen = generate_text(overlay, case.prompt, use_anchor=True)
            anchor_constraint = score_constraint(anchor_text_gen, case.anchor_group)
            anchor_score = anchor_constraint["constraint_score"]

        delta = anchor_score - base_score if cluster == "flat" else 0.0
        rescued = cluster == "flat" and base_score < 1.0 and anchor_score >= 1.0

        tail_retention = phase_metrics.get("tail_retention_ratio")
        print(f"  cluster={cluster} r1_ref={cluster_info['r1_reference']:.3f}"
              if cluster_info["r1_reference"] else f"  cluster={cluster} r1_ref=None",
              end="")
        print(f" | base={base_score:.0f}", end="")
        if cluster == "flat":
            print(f" anchor={anchor_score:.0f} delta={delta:+.0f}"
                  f" {'RESCUED' if rescued else ''}", end="")
        print(f" | tail_ret={tail_retention:.3f}" if tail_retention else "")

        result = {
            "name": case.name,
            "anchor_group": case.anchor_group,
            "anchor_class": case.anchor_class,
            "cluster": cluster,
            "route": cluster_info["route"],
            "r1_reference": cluster_info["r1_reference"],
            "delta_template_pair": cluster_info["delta_template_pair"],
            "phase_metrics": phase_metrics,
            "base_constraint_score": base_score,
            "anchor_constraint_score": anchor_score if cluster == "flat" else None,
            "delta": delta,
            "rescued": rescued,
            "base_text_preview": base_text[:200],
            "anchor_text_preview": anchor_text_gen[:200] if cluster == "flat" else None,
        }
        results.append(result)
        cluster_stats[cluster].append(result)

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY")
    print("=" * 70)

    summary: dict[str, Any] = {}
    for cl in ("mature", "template", "flat"):
        items = cluster_stats.get(cl, [])
        n = len(items)
        if n == 0:
            summary[cl] = {"count": 0}
            continue

        base_scores = [r["base_constraint_score"] for r in items]
        mean_base = float(np.mean(base_scores))

        rescued_count = sum(1 for r in items if r.get("rescued"))
        flat_failed = sum(1 for r in items if r["base_constraint_score"] < 1.0)
        rescue_rate = rescued_count / max(1, flat_failed) if cl == "flat" else None

        summary[cl] = {
            "count": n,
            "mean_base_constraint": round(mean_base, 3),
            "rescued": rescued_count,
            "base_failed": flat_failed,
            "rescue_rate": round(rescue_rate, 3) if rescue_rate is not None else None,
        }
        print(f"\n  {cl.upper()}: {n} cases, mean_base={mean_base:.3f}", end="")
        if cl == "flat":
            print(f", rescued={rescued_count}/{flat_failed}, "
                  f"rescue_rate={rescue_rate:.3f}" if rescue_rate is not None else "", end="")
        print()
        for r in items:
            tag = " ← RESCUED" if r.get("rescued") else ""
            print(f"    {r['name']:40s} base={r['base_constraint_score']:.0f}"
                  f" {f'anchor={r[\"anchor_constraint_score\"]:.0f}' if r.get('anchor_constraint_score') is not None else ''}"
                  f"{tag}")

    # --- Correlations ---
    tail_rets = [r["phase_metrics"]["tail_retention_ratio"]
                 for r in results if r["phase_metrics"].get("tail_retention_ratio") is not None]
    base_scores_all = [r["base_constraint_score"]
                       for r in results if r["phase_metrics"].get("tail_retention_ratio") is not None]
    rho_tail = spearman_correlation(tail_rets, base_scores_all)

    print(f"\n  CORRELATION: tail_retention_ratio vs base_constraint = "
          f"{rho_tail:.4f}" if rho_tail is not None else "  N/A")

    # --- Save ---
    ARCHIVE = ROOT / "archive"
    ARCHIVE.mkdir(exist_ok=True)
    slug = model_name.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out_path = ARCHIVE / f"{slug}_expanded_diagnostic_{anchor_profile}.json"

    payload = {
        "metadata": {
            "model_name": model_name,
            "anchor_profile": anchor_profile,
            "n_cases_total": len(cases),
            "n_cases_processed": len(results),
            "n_layers": n_layers,
            "reference_layers": reference_layers,
            "mature_r1_threshold": mature_threshold,
            "template_delta_threshold": template_threshold,
            "seed": SEED,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "cluster_summary": summary,
        "correlation": {
            "tail_retention_ratio_vs_base_constraint": rho_tail,
        },
        "cases": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[ExpandedDiag] Saved → {out_path}")
    print(f"[ExpandedDiag] Done. {len(results)} cases processed.")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABPT Expanded Diagnostic — все кейсы, 3 кластера, rescue stats"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--anchor-profile", default="medium",
                        choices=list(list_anchor_span_profiles()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mature-threshold", type=float, default=DEFAULT_MATURE_R1_THRESHOLD)
    parser.add_argument("--template-threshold", type=float, default=DEFAULT_TEMPLATE_DELTA_THRESHOLD)
    args, _ = parser.parse_known_args()

    run(
        model_name=args.model,
        anchor_profile=args.anchor_profile,
        device_str=args.device,
        mature_threshold=args.mature_threshold,
        template_threshold=args.template_threshold,
    )


if __name__ == "__main__":
    main()
