"""
ABPT Contradiction Proof Diagnostic
====================================
Детальная диагностика провала proof_by_contradiction кейсов (delta=-1).

Гипотезы для проверки:
  H5a: Carryover peak (L25) попадает в handoff zone (L24-31) → конфликт
  H5b: Template cluster неправильно классифицирует proof кейсы → trust route
       вместо anchor_forced
  H5c: Base generation уже корректна, anchor generation ломает её

Что делает:
  1. Извлекает полный r1 профиль для всех proof кейсов
  2. Сравнивает с кейсами других procedure_like групп
  3. Показывает layer-by-layer r1 для visual inspection
  4. Генерирует base и anchor текст для каждого кейса
  5. Анализирует carryover peak position vs handoff zone

Вывод:
  archive/qwen35_4b_contradiction_diagnostic.json

Использование:
  python scripts/run_qwen_contradiction_diagnostic.py
  python scripts/run_qwen_contradiction_diagnostic.py --device cpu
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
from scripts.run_qwen_phase_probe import (
    compute_phase_metrics,
    score_constraint,
    SEED,
    HANDOFF_START,
)

# Procedure groups to compare
PROOF_GROUP = "proof_by_contradiction_reasoning_steps"
INDUCTION_GROUP = "mathematical_induction_proof_steps"
COMPARISON_PROCEDURE_GROUPS = {
    "binary_search_update_loop_procedure",
    "dependency_injection_request_flow_sequence",
    "recursive_tree_traversal_procedure",
    "thread_safe_singleton_initialization_pattern",
}

MAX_LENGTH = 160
MAX_NEW_TOKENS = 120


def extract_r1_and_carryover(
    overlay: QwenAnchorOverlay,
    case: QwenAnchorGeometryCase,
    n_layers: int,
    device: torch.device,
) -> dict[str, Any] | None:
    tokenizer = overlay.tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer required")

    offsets = None
    try:
        encoded = tokenizer(
            case.prompt, truncation=True, max_length=MAX_LENGTH,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = [(int(s), int(e)) for s, e in encoded.pop("offset_mapping")[0].tolist()]
    except TypeError:
        encoded = tokenizer(
            case.prompt, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
        )

    batch = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    input_ids = [int(t) for t in batch["input_ids"][0].tolist()]

    span_match = match_anchor_span(
        text=case.prompt, anchor_text=case.anchor_text,
        input_ids=input_ids, tokenizer=tokenizer, offsets=offsets,
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

    # Full r1 profile
    r1_profile: dict[str, float] = {}
    for layer in range(n_layers):
        if layer + 1 >= len(hidden_states):
            continue
        delta_vecs = extract_delta_vectors(
            hidden_states[layer + 1][0],
            span_match.token_start,
            span_match.token_end,
        )
        metrics = compute_geometry_metrics(delta_vecs)
        val = metrics.get("rank1_explained_variance")
        r1_profile[str(layer)] = float(val) if val is not None else 0.0

    # Carryover: cosine similarity of suffix hidden state with anchor concept direction
    # Concept direction = mean of delta vectors at anchor span
    carryover_by_layer: dict[str, float] = {}
    for layer in range(n_layers):
        if layer + 1 >= len(hidden_states):
            continue
        hs = hidden_states[layer + 1][0]  # [seq_len, hidden_dim]
        anchor_vecs = hs[span_match.token_start:span_match.token_end + 1]
        if anchor_vecs.shape[0] == 0:
            continue
        concept_dir = anchor_vecs.mean(dim=0)
        concept_norm = concept_dir.norm()
        if concept_norm < 1e-8:
            continue

        # Suffix = tokens after anchor span
        suffix_start = span_match.token_end + 1
        if suffix_start >= hs.shape[0]:
            continue
        suffix_vecs = hs[suffix_start:]
        suffix_mean = suffix_vecs.mean(dim=0)
        cosine = float(torch.nn.functional.cosine_similarity(
            concept_dir.unsqueeze(0), suffix_mean.unsqueeze(0)
        ).item())
        carryover_by_layer[str(layer)] = cosine

    # Find carryover peak
    if carryover_by_layer:
        peak_layer = max(carryover_by_layer, key=carryover_by_layer.get)
        peak_value = carryover_by_layer[peak_layer]
        in_handoff = int(peak_layer) >= HANDOFF_START
    else:
        peak_layer = None
        peak_value = None
        in_handoff = None

    return {
        "r1_profile": r1_profile,
        "carryover_by_layer": carryover_by_layer,
        "carryover_peak_layer": int(peak_layer) if peak_layer else None,
        "carryover_peak_value": peak_value,
        "carryover_in_handoff_zone": in_handoff,
        "span_token_count": span_match.token_count,
    }


def generate_and_score(
    overlay: QwenAnchorOverlay,
    prompt: str,
    group: str,
    use_anchor: bool,
) -> dict[str, Any]:
    tokenizer = overlay.tokenizer
    device = next(overlay.parameters()).device
    encoded = tokenizer(
        [prompt], truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt", padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attn = encoded.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    n_prompt = int(input_ids.shape[1])

    if use_anchor and hasattr(overlay, "generate_with_anchor_bias"):
        generated = overlay.generate_with_anchor_bias(
            input_ids, attention_mask=attn, max_new_tokens=MAX_NEW_TOKENS,
        )
    else:
        with torch.no_grad():
            generated = overlay.base_model.generate(
                input_ids, attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None,
            )
    text = tokenizer.decode(generated[0][n_prompt:], skip_special_tokens=True)
    constraint = score_constraint(text, group)
    return {
        "text": text,
        "constraint_score": constraint["constraint_score"],
        "positive_hits": constraint["positive_hits"],
        "negative_hits": constraint["negative_hits"],
    }


def run(model_name: str, device_str: str) -> None:
    torch.manual_seed(SEED)
    device = torch.device(device_str)

    print(f"[ContraDiag] Model: {model_name}")
    print(f"[ContraDiag] Device: {device}")

    overlay = QwenAnchorOverlay.from_pretrained(model_name, config=TOY_CONFIG)
    overlay.to(device)
    overlay.eval()
    n_layers = int(overlay.model_num_hidden_layers)

    cases = make_qwen_anchor_geometry_cases(anchor_span_profile="medium")

    # Split into proof vs other-procedure
    proof_cases = [c for c in cases if c.anchor_group == PROOF_GROUP]
    induction_cases = [c for c in cases if c.anchor_group == INDUCTION_GROUP]
    other_procedure = [c for c in cases
                       if c.anchor_class == "procedure_like"
                       and c.anchor_group not in (PROOF_GROUP, INDUCTION_GROUP)]

    print(f"[ContraDiag] Proof cases: {len(proof_cases)}")
    print(f"[ContraDiag] Induction cases: {len(induction_cases)}")
    print(f"[ContraDiag] Other procedure cases: {len(other_procedure)}")

    all_results: list[dict[str, Any]] = []

    for label, case_set in [
        ("proof_contradiction", proof_cases),
        ("math_induction", induction_cases),
        ("other_procedure", other_procedure),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label}: {len(case_set)} cases")
        print(f"{'='*60}")

        for case in case_set:
            print(f"\n  {case.name}")

            geo = extract_r1_and_carryover(overlay, case, n_layers, device)
            if geo is None:
                print("    SKIP (span not found)")
                continue

            phase_metrics = compute_phase_metrics(geo["r1_profile"], n_layers)

            # Generate base and anchor
            base_result = generate_and_score(overlay, case.prompt, case.anchor_group, use_anchor=False)
            anchor_result = generate_and_score(overlay, case.prompt, case.anchor_group, use_anchor=True)

            delta = anchor_result["constraint_score"] - base_result["constraint_score"]

            print(f"    carryover_peak=L{geo['carryover_peak_layer']}"
                  f" ({'IN HANDOFF' if geo['carryover_in_handoff_zone'] else 'pre-handoff'})")
            print(f"    base={base_result['constraint_score']:.0f}"
                  f" anchor={anchor_result['constraint_score']:.0f}"
                  f" delta={delta:+.0f}")

            # r1 curve summary
            r1_vals = [geo["r1_profile"].get(str(l), 0.0) for l in range(n_layers)]
            r1_handoff = [geo["r1_profile"].get(str(l), 0.0) for l in range(HANDOFF_START, n_layers)]
            r1_crystal = [geo["r1_profile"].get(str(l), 0.0) for l in range(4, 9)]

            print(f"    r1_crystal_mean={np.mean(r1_crystal):.3f}"
                  f" r1_handoff_mean={np.mean(r1_handoff):.3f}"
                  f" tail_ret={phase_metrics.get('tail_retention_ratio', 'N/A')}")

            result = {
                "group_label": label,
                "name": case.name,
                "anchor_group": case.anchor_group,
                "carryover_peak_layer": geo["carryover_peak_layer"],
                "carryover_peak_value": geo["carryover_peak_value"],
                "carryover_in_handoff_zone": geo["carryover_in_handoff_zone"],
                "span_token_count": geo["span_token_count"],
                "phase_metrics": phase_metrics,
                "r1_crystal_mean": float(np.mean(r1_crystal)),
                "r1_handoff_mean": float(np.mean(r1_handoff)),
                "base_score": base_result["constraint_score"],
                "anchor_score": anchor_result["constraint_score"],
                "delta": delta,
                "base_positive_hits": base_result["positive_hits"],
                "anchor_positive_hits": anchor_result["positive_hits"],
                "base_negative_hits": base_result["negative_hits"],
                "anchor_negative_hits": anchor_result["negative_hits"],
                "base_preview": base_result["text"][:300],
                "anchor_preview": anchor_result["text"][:300],
                "r1_profile": geo["r1_profile"],
                "carryover_profile": geo["carryover_by_layer"],
            }
            all_results.append(result)

    # ─────────────────────────────────────────────────────────────────────
    # Analysis: test hypotheses
    # ─────────────��──────────────────────────────���────────────────────────

    print(f"\n{'='*60}")
    print("HYPOTHESIS TESTING")
    print(f"{'='*60}")

    proof_results = [r for r in all_results if r["group_label"] == "proof_contradiction"]
    other_results = [r for r in all_results if r["group_label"] == "other_procedure"]
    induction_results = [r for r in all_results if r["group_label"] == "math_induction"]

    # H5a: carryover in handoff zone
    proof_in_handoff = sum(1 for r in proof_results if r["carryover_in_handoff_zone"])
    other_in_handoff = sum(1 for r in other_results if r["carryover_in_handoff_zone"])
    print(f"\n  H5a (carryover in handoff zone → conflict):")
    print(f"    proof:     {proof_in_handoff}/{len(proof_results)} in handoff")
    print(f"    other:     {other_in_handoff}/{len(other_results)} in handoff")
    if induction_results:
        ind_in_handoff = sum(1 for r in induction_results if r["carryover_in_handoff_zone"])
        print(f"    induction: {ind_in_handoff}/{len(induction_results)} in handoff")

    # H5b: mean delta by group
    proof_deltas = [r["delta"] for r in proof_results]
    other_deltas = [r["delta"] for r in other_results]
    print(f"\n  H5b (delta comparison):")
    print(f"    proof mean delta:     {np.mean(proof_deltas):+.3f}" if proof_deltas else "    proof: N/A")
    print(f"    other mean delta:     {np.mean(other_deltas):+.3f}" if other_deltas else "    other: N/A")
    if induction_results:
        ind_deltas = [r["delta"] for r in induction_results]
        print(f"    induction mean delta: {np.mean(ind_deltas):+.3f}" if ind_deltas else "    induction: N/A")

    # H5c: base already correct?
    proof_base_ok = sum(1 for r in proof_results if r["base_score"] >= 1.0)
    print(f"\n  H5c (base already correct):")
    print(f"    proof base correct:   {proof_base_ok}/{len(proof_results)}")
    print(f"    → anchor BREAKS correct base" if proof_base_ok > 0 else "    → base also fails")

    # ─────────────────────────────────────────────────────────────────────
    # Save
    # ─��─────────────────────��────────────────────────��────────────────────

    ARCHIVE = ROOT / "archive"
    ARCHIVE.mkdir(exist_ok=True)
    slug = model_name.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out_path = ARCHIVE / f"{slug}_contradiction_diagnostic.json"

    payload = {
        "metadata": {
            "model_name": model_name,
            "n_layers": n_layers,
            "seed": SEED,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        "hypotheses": {
            "H5a_carryover_in_handoff": {
                "proof_in_handoff": proof_in_handoff,
                "proof_total": len(proof_results),
                "other_in_handoff": other_in_handoff,
                "other_total": len(other_results),
            },
            "H5b_delta_comparison": {
                "proof_mean_delta": float(np.mean(proof_deltas)) if proof_deltas else None,
                "other_mean_delta": float(np.mean(other_deltas)) if other_deltas else None,
            },
            "H5c_base_already_correct": {
                "proof_base_correct": proof_base_ok,
                "proof_total": len(proof_results),
            },
        },
        "cases": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[ContraDiag] Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ABPT Contradiction Proof Diagnostic")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args()
    run(model_name=args.model, device_str=args.device)


if __name__ == "__main__":
    main()
