from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import re
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import (
    list_anchor_span_profiles,
    make_qwen_anchor_geometry_cases,
)
from src.data.qwen_anchor_neutral_cases import make_qwen_anchor_neutral_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.qwen_anchor_cartography import (
    SpanEncoding,
    build_group_concept_vectors,
    compute_neutral_basis,
    cosine_or_none,
    encode_focus_span,
    project_out_basis,
)
from src.utils.anchor_geometry import list_model_layers


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "item"


def _token_vectors_by_layer(
    *,
    encoding: SpanEncoding,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    output: dict[int, torch.Tensor] = {}
    for layer in layers:
        hidden_state_index = int(layer) + 1
        output[int(layer)] = encoding.hidden_states[hidden_state_index][0].to(dtype=torch.float32).detach().cpu()
    return output


def _concept_pairwise_cosines(
    *,
    concept_vectors: dict[int, dict[str, torch.Tensor]],
) -> dict[int, dict[str, dict[str, float | None]]]:
    payload: dict[int, dict[str, dict[str, float | None]]] = {}
    for layer, group_map in concept_vectors.items():
        groups = sorted(group_map)
        payload[int(layer)] = {
            left: {
                right: cosine_or_none(group_map[left], group_map[right])
                for right in groups
            }
            for left in groups
        }
    return payload


def plot_case_heatmap(
    *,
    case: QwenAnchorGeometryCase,
    encoding: SpanEncoding,
    layers: list[int],
    group_vectors: dict[int, dict[str, torch.Tensor]],
    neutral_basis_by_layer: dict[int, torch.Tensor | None],
    output_path: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    token_vectors = _token_vectors_by_layer(encoding=encoding, layers=layers)
    heat = []
    for layer in layers:
        basis = neutral_basis_by_layer.get(int(layer))
        projected_tokens = []
        for token_vector in token_vectors[int(layer)]:
            projected_token = project_out_basis(token_vector, basis)
            projected_tokens.append(cosine_or_none(projected_token, group_vectors[int(layer)][case.anchor_group]))
        heat.append([value if value is not None else float("nan") for value in projected_tokens])
    matrix = np.array(heat, dtype=float)
    fig, ax = plt.subplots(figsize=(16, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(f"{case.name} | {case.anchor_group}")
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{int(layer)}" for layer in layers])
    span_start = int(encoding.span_match.token_start)
    span_end = int(encoding.span_match.token_end)
    ax.axvline(span_start - 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axvline(span_end + 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="cosine to concept vector")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    peak_index = int(torch.tensor(matrix, dtype=torch.float32).nan_to_num(-2.0).argmax().item())
    peak_layer_idx = peak_index // matrix.shape[1]
    peak_token_idx = peak_index % matrix.shape[1]
    peak_value = matrix[peak_layer_idx, peak_token_idx]
    return {
        "peak_layer": int(layers[peak_layer_idx]),
        "peak_token_index": int(peak_token_idx),
        "peak_cosine": None if peak_value != peak_value else float(peak_value),
        "anchor_span_start": span_start,
        "anchor_span_end": span_end,
    }


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    profile_payloads: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen Anchor Concept Direction Map",
        "",
        "## Summary",
        "",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "",
        "Этот прогон строит concept vectors по anchor groups, вычитает общую среднюю активацию и проектирует их из neutral subspace, а затем смотрит cosine-карты `token × layer` для каждого кейса.",
        "",
    ]
    for profile in profile_payloads:
        lines.extend(
            [
                f"## Profile: `{profile['profile']}`",
                "",
                "| case | class | group | peak layer | peak token index | peak cosine | figure |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for case_payload in profile["cases"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        case_payload["name"],
                        case_payload["anchor_class"],
                        case_payload["anchor_group"],
                        str(case_payload["heatmap_summary"]["peak_layer"]),
                        str(case_payload["heatmap_summary"]["peak_token_index"]),
                        f"{float(case_payload['heatmap_summary']['peak_cosine']):.3f}" if case_payload["heatmap_summary"]["peak_cosine"] is not None else "n/a",
                        f"[plot]({case_payload['figure_relpath']})",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Layerwise concept vector norms",
                "",
                "| layer | " + " | ".join(profile["groups"]) + " |",
                "| --- | " + " | ".join(["---:"] * len(profile["groups"])) + " |",
            ]
        )
        for layer_entry in profile["concept_vector_norms"]:
            lines.append(
                "| "
                + " | ".join(
                    [f"L{int(layer_entry['layer']):02d}"]
                    + [
                        f"{float(layer_entry[group]):.3f}" if layer_entry[group] is not None else "n/a"
                        for group in profile["groups"]
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build concept-vector direction maps for Qwen anchor groups.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--neutral_components", type=int, default=3)
    parser.add_argument("--neutral_variance_cutoff", type=float, default=0.5)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(list_anchor_span_profiles()),
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_anchor_concept_direction_map.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_concept_direction_map.md",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=ROOT / "docs" / "research" / "figures" / "qwen_anchor_concept_direction_map",
    )
    args = parser.parse_args()

    for profile in args.profiles:
        if profile not in list_anchor_span_profiles():
            raise ValueError(f"unknown profile: {profile}")

    device = torch.device(args.device)
    overlay = QwenAnchorOverlay.from_pretrained(
        model_name=args.model,
        cfg=TOY_CONFIG,
        device=device,
        torch_dtype=torch.float16 if "cuda" in str(device) else None,
        low_cpu_mem_usage=True,
    )
    overlay.eval()
    layers = list_model_layers(int(getattr(overlay, "model_num_hidden_layers", 0)))
    neutral_cases = make_qwen_anchor_neutral_cases()
    neutral_encodings = {
        case.name: encode_focus_span(
            overlay=overlay,
            text=case.prompt,
            focus_text=case.focus_text,
            max_length=int(args.max_length),
            device=device,
        )
        for case in neutral_cases
    }
    if any(value is None for value in neutral_encodings.values()):
        missing = [name for name, value in neutral_encodings.items() if value is None]
        raise ValueError(f"neutral span matching failed for cases: {missing}")

    neutral_basis_by_layer: dict[int, torch.Tensor | None] = {}
    for layer in layers:
        neutral_vectors = [
            span_mean_hidden(
                neutral_encodings[case.name].hidden_states[int(layer) + 1][0],
                token_start=int(neutral_encodings[case.name].span_match.token_start),
                token_end=int(neutral_encodings[case.name].span_match.token_end),
            )
            for case in neutral_cases
        ]
        neutral_basis_by_layer[int(layer)] = compute_neutral_basis(
            neutral_vectors,
            max_components=int(args.neutral_components),
            variance_cutoff=float(args.neutral_variance_cutoff),
        )

    profile_payloads: list[dict[str, Any]] = []
    for profile_name in args.profiles:
        cases = make_qwen_anchor_geometry_cases(anchor_span_profile=profile_name)
        if args.case_name:
            cases = [case for case in cases if case.name == args.case_name]
        if args.limit is not None:
            cases = cases[: max(int(args.limit), 0)]
        if not cases:
            raise ValueError(f"no cases selected for profile {profile_name}")
        encodings = {
            case.name: encode_focus_span(
                overlay=overlay,
                text=case.prompt,
                focus_text=case.anchor_text,
                max_length=int(args.max_length),
                device=device,
            )
            for case in cases
        }
        if any(value is None for value in encodings.values()):
            missing = [name for name, value in encodings.items() if value is None]
            raise ValueError(f"anchor span matching failed for profile {profile_name}: {missing}")
        concept_vectors, concept_norms = build_group_concept_vectors(
            layers=layers,
            name_to_group={case.name: case.anchor_group for case in cases},
            encodings=encodings,
            neutral_basis_by_layer=neutral_basis_by_layer,
        )
        pairwise_cosines = _concept_pairwise_cosines(concept_vectors=concept_vectors)
        groups = sorted({case.anchor_group for case in cases})
        case_payloads: list[dict[str, Any]] = []
        for case in cases:
            encoding = encodings[case.name]
            figure_name = f"{_slug(profile_name)}_{_slug(case.name)}.png"
            figure_path = args.figure_dir / figure_name
            heatmap_summary = plot_case_heatmap(
                case=case,
                encoding=encoding,
                layers=layers,
                group_vectors=concept_vectors,
                neutral_basis_by_layer=neutral_basis_by_layer,
                output_path=figure_path,
            )
            case_payloads.append(
                {
                    **asdict(case),
                    "match_method": str(encoding.span_match.match_method),
                    "token_count": int(encoding.span_match.token_count),
                    "heatmap_summary": heatmap_summary,
                    "figure_path": str(figure_path),
                    "figure_relpath": Path(
                        Path("figures") / "qwen_anchor_concept_direction_map" / figure_name
                    ).as_posix(),
                }
            )
        profile_payloads.append(
            {
                "profile": profile_name,
                "groups": groups,
                "cases": case_payloads,
                "concept_vector_norms": [
                    {"layer": int(layer), **concept_norms[int(layer)]}
                    for layer in layers
                ],
                "concept_pairwise_cosines": pairwise_cosines,
            }
        )

    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": args.model,
            "device": str(device),
            "max_length": int(args.max_length),
            "probe_layers": layers,
            "profiles": list(args.profiles),
            "neutral_components": int(args.neutral_components),
            "neutral_variance_cutoff": float(args.neutral_variance_cutoff),
        },
        "profiles": profile_payloads,
    }
    report = build_markdown_report(
        model_name=args.model,
        device=str(device),
        profile_payloads=profile_payloads,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved_json={args.output_json}")
    print(f"saved_md={args.output_md}")
    print(f"saved_figures={args.figure_dir}")


if __name__ == "__main__":
    main()
