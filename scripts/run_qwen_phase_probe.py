"""
ABPT Phase Probe — Фаза 1 Верификации
======================================
Считает 8 геометрических метрик зоны кристаллизации L4-L8 для каждого кейса,
запускает base генерацию, собирает constraint_score.
Затем вычисляет корреляцию Спирмена между early_slope_4_8 и base_constraint_score.

Метрики:
  early_slope_4_8         — наклон r1 между L4 и L8 (linreg)
  early_auc_4_8           — площадь под кривой r1 в [L4..L8]
  peak_layer_4_12         — слой с max r1 в [L4..L12]
  peak_value_4_12         — значение max r1 в [L4..L12]
  profile_width_above_tau — число слоёв где r1 >= tau (default tau=0.5)
  sharpness               — peak_value / max(1, width)
  tail_retention_ratio    — auc(L9..L23) / auc(L4..L8)
  late_decay_ratio        — auc(L24..L31) / auc(L4..L8)

Использование (в Colab):
  !python scripts/run_qwen_phase_probe.py \\
      --model Qwen/Qwen3.5-4B \\
      --anchor-profile medium \\
      --seed 7

Вывод:
  - JSON: archive/qwen35_4b_phase_probe_<profile>.json
  - MD:   docs/research/qwen35_4b_phase_probe_<profile>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
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
    compute_geometry_metrics,
    extract_delta_vectors,
    list_model_layers,
    select_tail_probe_layers,
)
from src.utils.anchor_geometry import (
    decode_token_pieces,
    decode_token_surfaces,
    match_anchor_span,
    token_has_leading_whitespace,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CRYSTALLIZATION_START = 4
CRYSTALLIZATION_END   = 8
PROPAGATION_START     = 9
PROPAGATION_END       = 15
INTEGRATION_START     = 16
INTEGRATION_END       = 23
HANDOFF_START         = 24

DEFAULT_TAU           = 0.50   # r1 threshold for "above_tau" width
MAX_LENGTH            = 160
MAX_NEW_TOKENS        = 120
SEED                  = 7

# keyword specs per group (positive = нужны, negative = запрещены)
KEYWORD_SPECS: dict[str, dict[str, Any]] = {
    "strictly_vegan_meal_plan_policy": {
        "positive": ["vegan", "plant-based", "plant based", "dairy-free",
                     "egg-free", "animal-free"],
        "negative": ["meat", "chicken", "beef", "pork", "fish", "dairy",
                     "milk", "cheese", "butter", "egg", "eggs"],
        "negative_exceptions": {},
        "min_unique_positive_hits": 2,
    },
    "async_fastapi_service_architecture_policy": {
        "positive": ["async", "await", "fastapi", "asyncio"],
        "negative": ["flask", "django", "synchronous", "sync def"],
        "negative_exceptions": {},
        "min_unique_positive_hits": 2,
    },
    "json_only_response_format_policy": {
        "positive": ["json", "{", "}"],
        "negative": ["here is", "sure", "```python", "explanation"],
        "negative_exceptions": {},
        "min_unique_positive_hits": 2,
    },
    "proof_by_contradiction_reasoning_steps": {
        "positive": ["assume", "contradiction", "therefore", "suppose",
                     "hence", "negation"],
        "negative": [],
        "negative_exceptions": {},
        "min_unique_positive_hits": 2,
    },
    "binary_search_update_loop_procedure": {
        "positive": ["low", "high", "mid", "while", "binary search"],
        "negative": ["linear search", "sequential"],
        "negative_exceptions": {},
        "min_unique_positive_hits": 2,
    },
    "dependency_injection_request_flow_sequence": {
        "positive": ["inject", "dependency", "container", "provider",
                     "resolve", "service"],
        "negative": [],
        "negative_exceptions": {},
        "min_unique_positive_hits": 2,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_scalar(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _auc(r1_profile: dict[str, float | None], start: int, end: int) -> float:
    """Трапецоидная интеграция r1 по слоям [start..end]."""
    layers = list(range(start, end + 1))
    vals = [r1_profile.get(str(l)) for l in layers]
    vals = [v if v is not None else 0.0 for v in vals]
    if len(vals) < 2:
        return float(vals[0]) if vals else 0.0
    return float(np.trapz(vals))


def compute_phase_metrics(
    r1_profile: dict[str, float | None],
    n_layers: int,
    tau: float = DEFAULT_TAU,
) -> dict[str, float | None]:
    cs = CRYSTALLIZATION_START
    ce = CRYSTALLIZATION_END
    handoff_end = n_layers - 1

    # early slope: linreg через L4..L8
    xs = np.array(list(range(cs, ce + 1)), dtype=np.float64)
    ys = np.array(
        [float(r1_profile.get(str(l)) or 0.0) for l in range(cs, ce + 1)],
        dtype=np.float64,
    )
    if len(xs) >= 2:
        early_slope = float(np.polyfit(xs, ys, deg=1)[0])
    else:
        early_slope = None

    # early auc L4..L8
    early_auc = _auc(r1_profile, cs, ce)

    # peak в L4..L12
    peak_search_end = min(12, n_layers - 1)
    peak_val: float | None = None
    peak_layer: int | None = None
    for l in range(cs, peak_search_end + 1):
        v = _to_scalar(r1_profile.get(str(l)))
        if v is not None and (peak_val is None or v > peak_val):
            peak_val = v
            peak_layer = l

    # width above tau (all layers)
    width = sum(
        1 for l in range(n_layers)
        if (_to_scalar(r1_profile.get(str(l))) or 0.0) >= tau
    )

    sharpness = (peak_val / max(1, width)) if peak_val is not None else None

    # tail retention: auc(L9..L23) / auc(L4..L8)
    tail_auc = _auc(r1_profile, PROPAGATION_START, INTEGRATION_END)
    tail_retention = (tail_auc / early_auc) if early_auc > 0 else None

    # late decay: auc(L24..end) / auc(L4..L8)
    late_auc = _auc(r1_profile, HANDOFF_START, handoff_end)
    late_decay = (late_auc / early_auc) if early_auc > 0 else None

    return {
        "early_slope_4_8":       early_slope,
        "early_auc_4_8":         early_auc,
        "peak_layer_4_12":       peak_layer,
        "peak_value_4_12":       peak_val,
        "profile_width_above_tau": float(width),
        "tau":                   tau,
        "sharpness":             sharpness,
        "tail_retention_ratio":  tail_retention,
        "late_decay_ratio":      late_decay,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry extraction (повторяет логику calibration script)
# ─────────────────────────────────────────────────────────────────────────────

def extract_rank1_profile(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    probe_layers: list[int],
    device: torch.device,
) -> dict[str, float | None] | None:
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is required")

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
        offsets = None

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
        print(f"  [WARN] span not found for case '{case.name}'")
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
        delta_vecs = extract_delta_vectors(
            hidden_states[layer + 1][0],
            span_match.token_start,
            span_match.token_end,
        )
        metrics = compute_geometry_metrics(delta_vecs)
        r1_profile[str(layer)] = _to_scalar(metrics.get("rank1_explained_variance"))

    return r1_profile


# ─────────────────────────────────────────────────────────────────────────────
# Base generation + constraint scoring
# ─────────────────────────────────────────────────────────────────────────────

def generate_base_text(
    overlay: QwenAnchorOverlay,
    prompt: str,
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

    n_prompt_tokens = int(input_ids.shape[1])
    with torch.no_grad():
        generated = overlay.base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    continuation = generated[0][n_prompt_tokens:]
    return tokenizer.decode(continuation, skip_special_tokens=True)


def score_constraint(text: str, group: str) -> dict[str, Any]:
    spec = KEYWORD_SPECS.get(group, {})
    positive = [t.lower() for t in spec.get("positive", [])]
    negative = [t.lower() for t in spec.get("negative", [])]
    neg_exc = {
        k.lower(): [p.lower() for p in v]
        for k, v in spec.get("negative_exceptions", {}).items()
    }
    min_pos = int(spec.get("min_unique_positive_hits", 2))
    lowered = text.lower()

    pos_hits = {t: lowered.count(t) for t in positive if lowered.count(t) > 0}
    neg_hits: dict[str, int] = {}
    for t in negative:
        exc_phrases = neg_exc.get(t, [])
        count = lowered.count(t)
        if exc_phrases:
            protected = sum(lowered.count(p) for p in exc_phrases)
            count = max(0, count - protected)
        if count > 0:
            neg_hits[t] = count

    unique_pos = len(pos_hits)
    neg_total = sum(neg_hits.values())
    satisfied = (unique_pos >= min_pos) and (neg_total == 0)

    return {
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "unique_positive_hits": unique_pos,
        "negative_total": neg_total,
        "constraint_satisfied": satisfied,
        "constraint_score": 1.0 if satisfied else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────────────────────────────────────

def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(xs, ys)
        return float(rho)
    except ImportError:
        # Ручная реализация
        n = len(xs)
        rx = _rank(xs)
        ry = _rank(ys)
        d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
        return 1.0 - (6 * d2) / (n * (n ** 2 - 1))


def _rank(vals: list[float]) -> list[float]:
    sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    for rank, (original_idx, _) in enumerate(sorted_vals, start=1):
        ranks[original_idx] = float(rank)
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    model_name: str,
    anchor_profile: str,
    tau: float,
    device_str: str,
) -> None:
    torch.manual_seed(SEED)
    device = torch.device(device_str)

    print(f"[PhaseProbe] Model:   {model_name}")
    print(f"[PhaseProbe] Profile: {anchor_profile}")
    print(f"[PhaseProbe] Device:  {device}")

    print("[PhaseProbe] Загружаю модель...")
    overlay = QwenAnchorOverlay.from_pretrained(model_name, config=TOY_CONFIG)
    overlay.to(device)
    overlay.eval()

    n_layers = overlay.base_model.config.num_hidden_layers
    probe_layers = list(range(n_layers))
    print(f"[PhaseProbe] Слоёв: {n_layers}")

    cases = make_qwen_anchor_geometry_cases(anchor_span_profile=anchor_profile)
    print(f"[PhaseProbe] Кейсов: {len(cases)}")

    results = []
    slopes_for_corr: list[float] = []
    constraints_for_corr: list[float] = []

    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case.name}")

        # --- Геометрия ---
        r1_profile = extract_rank1_profile(overlay, case, probe_layers, device)
        if r1_profile is None:
            print("  SKIP (span not found)")
            continue

        phase_metrics = compute_phase_metrics(r1_profile, n_layers, tau=tau)

        # --- Base generation ---
        print("  generating base...")
        base_text = generate_base_text(overlay, case.prompt)
        constraint = score_constraint(base_text, case.anchor_group)
        base_constraint_score = constraint["constraint_score"]

        print(f"  early_slope_4_8 = {phase_metrics['early_slope_4_8']:.4f}"
              f"  base_constraint = {base_constraint_score:.0f}")
        print(f"  peak@L{phase_metrics['peak_layer_4_12']} = {phase_metrics['peak_value_4_12']:.3f}"
              f"  sharpness = {phase_metrics['sharpness']:.3f}"
              if phase_metrics['peak_value_4_12'] else "  (no peak)")

        result = {
            "name": case.name,
            "anchor_group": case.anchor_group,
            "anchor_class": case.anchor_class,
            "anchor_text": case.anchor_text,
            "r1_profile": r1_profile,
            "phase_metrics": phase_metrics,
            "base_generated_text": base_text,
            "base_constraint": constraint,
            "base_constraint_score": base_constraint_score,
        }
        results.append(result)

        slope = phase_metrics.get("early_slope_4_8")
        if slope is not None:
            slopes_for_corr.append(slope)
            constraints_for_corr.append(base_constraint_score)

    # --- Корреляция ---
    rho = spearman_correlation(slopes_for_corr, constraints_for_corr)
    print(f"\n[PhaseProbe] Spearman ρ (early_slope_4_8 vs base_constraint_score) = "
          f"{rho:.4f}" if rho is not None else "N/A (< 3 points)")

    # Корреляции для всех метрик
    metric_correlations: dict[str, float | None] = {}
    metric_names = [
        "early_slope_4_8", "early_auc_4_8", "peak_value_4_12",
        "sharpness", "tail_retention_ratio", "late_decay_ratio",
    ]
    for mname in metric_names:
        mvals = [
            r["phase_metrics"].get(mname)
            for r in results
            if r["phase_metrics"].get(mname) is not None
        ]
        cvals = [
            r["base_constraint_score"]
            for r in results
            if r["phase_metrics"].get(mname) is not None
        ]
        metric_correlations[mname] = spearman_correlation(mvals, cvals)

    # --- Сохранение JSON ---
    ARCHIVE = ROOT / "archive"
    ARCHIVE.mkdir(exist_ok=True)
    slug = model_name.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out_json = ARCHIVE / f"{slug}_phase_probe_{anchor_profile}.json"

    payload = {
        "metadata": {
            "model_name": model_name,
            "anchor_profile": anchor_profile,
            "tau": tau,
            "n_cases": len(results),
            "n_layers": n_layers,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "crystallization_zone": [CRYSTALLIZATION_START, CRYSTALLIZATION_END],
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        "correlation_summary": {
            "spearman_early_slope_4_8_vs_base_constraint": rho,
            "all_metrics": metric_correlations,
        },
        "cases": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[PhaseProbe] JSON → {out_json}")

    # --- Сохранение MD ---
    DOCS = ROOT / "docs" / "research"
    DOCS.mkdir(parents=True, exist_ok=True)
    out_md = DOCS / f"{slug}_phase_probe_{anchor_profile}.md"

    md_lines = [
        f"# ABPT Phase Probe — {model_name} / profile={anchor_profile}",
        f"",
        f"**Created:** {payload['metadata']['created_at_utc']}  ",
        f"**Cases:** {len(results)} | **Layers:** {n_layers} | **tau:** {tau}",
        f"",
        f"## Correlation Summary",
        f"",
        f"| Metric | Spearman ρ vs base_constraint_score |",
        f"|--------|--------------------------------------|",
    ]
    for mname, corr_val in metric_correlations.items():
        val_str = f"{corr_val:.4f}" if corr_val is not None else "N/A"
        md_lines.append(f"| `{mname}` | {val_str} |")

    md_lines += [
        f"",
        f"## Per-Case Results",
        f"",
        f"| name | group | early_slope | peak@L | peak_val | sharpness | tail_retention | base_constraint |",
        f"|------|-------|-------------|--------|----------|-----------|----------------|-----------------|",
    ]

    def _f(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    for r in results:
        pm = r["phase_metrics"]
        md_lines.append(
            f"| {r['name']} | {r['anchor_group'].split('_')[0]} "
            f"| {_f(pm.get('early_slope_4_8'))} "
            f"| L{pm.get('peak_layer_4_12')} "
            f"| {_f(pm.get('peak_value_4_12'))} "
            f"| {_f(pm.get('sharpness'))} "
            f"| {_f(pm.get('tail_retention_ratio'))} "
            f"| {_f(r['base_constraint_score'])} |"
        )

    md_lines += [
        f"",
        f"## Hypothesis",
        f"",
        f"Если `early_slope_4_8` **отрицательный или близкий к нулю** → r1 не растёт в зоне кристаллизации",
        f"→ base модель не сформировала устойчивую концептуальную структуру → высокий риск провала constraint.",
        f"",
        f"Если корреляция Spирмена **|ρ| > 0.4** — гипотеза подтверждена,",
        f"можно использовать `early_slope_4_8` как routing signal в Фазе 2.",
    ]

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[PhaseProbe] MD   → {out_md}")
    print(f"\n[PhaseProbe] Готово. ρ = {rho:.4f}" if rho is not None else "\n[PhaseProbe] Готово. ρ = N/A")


def main() -> None:
    parser = argparse.ArgumentParser(description="ABPT Phase Probe — Фаза 1 верификации геометрии")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B",
                        help="HuggingFace model name")
    parser.add_argument("--anchor-profile", default="medium",
                        choices=list(list_anchor_span_profiles()),
                        help="Профиль длины anchor span")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help="Порог r1 для подсчёта profile_width_above_tau")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run(
        model_name=args.model,
        anchor_profile=args.anchor_profile,
        tau=args.tau,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
