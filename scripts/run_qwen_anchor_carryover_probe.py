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

from src.data.qwen_anchor_carryover_cases import make_qwen_anchor_carryover_cases
from src.data.qwen_anchor_geometry_cases import list_anchor_span_profiles, make_qwen_anchor_geometry_cases
from src.data.qwen_anchor_neutral_cases import make_qwen_anchor_neutral_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import list_model_layers
from src.utils.qwen_anchor_cartography import (
    SpanEncoding,
    build_group_concept_vectors,
    compute_neutral_basis,
    cosine_or_none,
    encode_focus_span,
    project_out_basis,
    span_mean_hidden_for_layer,
)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "item"


def _suffix_cosine_matrix(
    *,
    encoding: SpanEncoding,
    layers: list[int],
    concept_vectors: dict[int, dict[str, torch.Tensor]],
    group_name: str,
    neutral_basis_by_layer: dict[int, torch.Tensor | None],
) -> list[list[float | None]]:
    matrix: list[list[float | None]] = []
    token_start = int(encoding.span_match.token_start)
    token_end = int(encoding.span_match.token_end)
    for layer in layers:
        layer_hidden = encoding.hidden_states[int(layer) + 1][0].to(dtype=torch.float32).detach().cpu()
        basis = neutral_basis_by_layer.get(int(layer))
        group_vector = concept_vectors[int(layer)].get(group_name)
        row: list[float | None] = []
        for token_index in range(token_start, token_end + 1):
            projected = project_out_basis(layer_hidden[token_index], basis)
            row.append(cosine_or_none(projected, group_vector))
        matrix.append(row)
    return matrix


def summarize_delta_matrix(
    *,
    layers: list[int],
    anchored_matrix: list[list[float | None]],
    neutral_matrix: list[list[float | None]],
) -> dict[str, Any]:
    if not anchored_matrix or not anchored_matrix[0]:
        return {
            "peak_delta_layer": None,
            "peak_delta_token_index": None,
            "peak_delta_value": None,
            "mean_delta_last_token": None,
            "layer_mean_deltas": [],
        }
    peak_value: float | None = None
    peak_layer: int | None = None
    peak_token_index: int | None = None
    layer_mean_deltas: list[dict[str, float | int | None]] = []
    last_token_values: list[float] = []
    for row_index, layer in enumerate(layers):
        row_deltas: list[float] = []
        for col_index, anchored_value in enumerate(anchored_matrix[row_index]):
            neutral_value = neutral_matrix[row_index][col_index]
            if anchored_value is None or neutral_value is None:
                continue
            delta = float(anchored_value - neutral_value)
            row_deltas.append(delta)
            if peak_value is None or delta > peak_value:
                peak_value = delta
                peak_layer = int(layer)
                peak_token_index = int(col_index)
        if row_deltas:
            layer_mean = float(sum(row_deltas) / len(row_deltas))
            last_token_values.append(float(row_deltas[-1]))
        else:
            layer_mean = None
        layer_mean_deltas.append({"layer": int(layer), "mean_delta": layer_mean})
    mean_delta_last_token = float(sum(last_token_values) / len(last_token_values)) if last_token_values else None
    return {
        "peak_delta_layer": peak_layer,
        "peak_delta_token_index": peak_token_index,
        "peak_delta_value": peak_value,
        "mean_delta_last_token": mean_delta_last_token,
        "layer_mean_deltas": layer_mean_deltas,
    }


def plot_carryover_case(
    *,
    case_name: str,
    group_name: str,
    layers: list[int],
    anchored_matrix: list[list[float | None]],
    neutral_matrix: list[list[float | None]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    anchored = np.array(
        [[float("nan") if value is None else float(value) for value in row] for row in anchored_matrix],
        dtype=float,
    )
    neutral = np.array(
        [[float("nan") if value is None else float(value) for value in row] for row in neutral_matrix],
        dtype=float,
    )
    delta = anchored - neutral

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, matrix, title in zip(
        axes,
        (anchored, neutral, delta),
        ("anchored prefix", "neutral prefix", "anchored - neutral"),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("shared suffix token")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{int(layer)}" for layer in layers])
    axes[0].set_ylabel("layer")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="cosine to concept vector")
    fig.suptitle(f"{case_name} | {group_name}", y=1.02, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(
    *,
    model_name: str,
    device: str,
    profile_payloads: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen Anchor Carryover Probe",
        "",
        "## Summary",
        "",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "",
        "Этот прогон сравнивает anchored prefix и neutral prefix при одном и том же shared suffix. Для каждого слоя считается, насколько suffix tokens ближе к group concept vector при наличии anchor-префикса.",
        "",
    ]
    for profile in profile_payloads:
        lines.extend(
            [
                f"## Profile: `{profile['profile']}`",
                "",
                "| case | group | peak delta layer | peak delta token | peak delta value | mean delta on last suffix token | figure |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for case_payload in profile["cases"]:
            summary = case_payload["delta_summary"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        case_payload["name"],
                        case_payload["anchor_group"],
                        str(summary["peak_delta_layer"] if summary["peak_delta_layer"] is not None else "n/a"),
                        str(summary["peak_delta_token_index"] if summary["peak_delta_token_index"] is not None else "n/a"),
                        f"{float(summary['peak_delta_value']):.3f}" if summary["peak_delta_value"] is not None else "n/a",
                        f"{float(summary['mean_delta_last_token']):.3f}" if summary["mean_delta_last_token"] is not None else "n/a",
                        f"[plot]({case_payload['figure_relpath']})",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe anchor carryover into a shared suffix.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--neutral_components", type=int, default=3)
    parser.add_argument("--neutral_variance_cutoff", type=float, default=0.5)
    parser.add_argument("--profiles", nargs="+", default=list(list_anchor_span_profiles()))
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "archive" / "qwen_anchor_carryover_probe.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "research" / "qwen_anchor_carryover_probe.md",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=ROOT / "docs" / "research" / "figures" / "qwen_anchor_carryover_probe",
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
            span_mean_hidden_for_layer(neutral_encodings[case.name], layer=int(layer))
            for case in neutral_cases
        ]
        neutral_basis_by_layer[int(layer)] = compute_neutral_basis(
            neutral_vectors,
            max_components=int(args.neutral_components),
            variance_cutoff=float(args.neutral_variance_cutoff),
        )

    profile_payloads: list[dict[str, Any]] = []
    for profile_name in args.profiles:
        geometry_cases = make_qwen_anchor_geometry_cases(anchor_span_profile=profile_name)
        geometry_encodings = {
            case.name: encode_focus_span(
                overlay=overlay,
                text=case.prompt,
                focus_text=case.anchor_text,
                max_length=int(args.max_length),
                device=device,
            )
            for case in geometry_cases
        }
        if any(value is None for value in geometry_encodings.values()):
            missing = [name for name, value in geometry_encodings.items() if value is None]
            raise ValueError(f"anchor span matching failed for geometry profile {profile_name}: {missing}")
        concept_vectors, concept_norms = build_group_concept_vectors(
            layers=layers,
            name_to_group={case.name: case.anchor_group for case in geometry_cases},
            encodings=geometry_encodings,
            neutral_basis_by_layer=neutral_basis_by_layer,
        )

        carryover_cases = make_qwen_anchor_carryover_cases(anchor_span_profile=profile_name)
        if args.case_name:
            carryover_cases = [case for case in carryover_cases if case.name == args.case_name]
        if not carryover_cases:
            raise ValueError(f"no carryover cases selected for profile {profile_name}")

        case_payloads: list[dict[str, Any]] = []
        for case in carryover_cases:
            anchored_text = f"{case.anchored_prefix}{case.shared_suffix}"
            neutral_text = f"{case.neutral_prefix}{case.shared_suffix}"
            anchored_encoding = encode_focus_span(
                overlay=overlay,
                text=anchored_text,
                focus_text=case.shared_suffix.strip(),
                max_length=int(args.max_length),
                device=device,
            )
            neutral_encoding = encode_focus_span(
                overlay=overlay,
                text=neutral_text,
                focus_text=case.shared_suffix.strip(),
                max_length=int(args.max_length),
                device=device,
            )
            if anchored_encoding is None or neutral_encoding is None:
                raise ValueError(f"shared suffix span matching failed for {case.name} ({profile_name})")
            anchored_matrix = _suffix_cosine_matrix(
                encoding=anchored_encoding,
                layers=layers,
                concept_vectors=concept_vectors,
                group_name=case.anchor_group,
                neutral_basis_by_layer=neutral_basis_by_layer,
            )
            neutral_matrix = _suffix_cosine_matrix(
                encoding=neutral_encoding,
                layers=layers,
                concept_vectors=concept_vectors,
                group_name=case.anchor_group,
                neutral_basis_by_layer=neutral_basis_by_layer,
            )
            delta_summary = summarize_delta_matrix(
                layers=layers,
                anchored_matrix=anchored_matrix,
                neutral_matrix=neutral_matrix,
            )
            figure_name = f"{_slug(profile_name)}_{_slug(case.name)}.png"
            figure_path = args.figure_dir / figure_name
            plot_carryover_case(
                case_name=case.name,
                group_name=case.anchor_group,
                layers=layers,
                anchored_matrix=anchored_matrix,
                neutral_matrix=neutral_matrix,
                output_path=figure_path,
            )
            case_payloads.append(
                {
                    **asdict(case),
                    "anchored_text": anchored_text,
                    "neutral_text": neutral_text,
                    "anchored_suffix_token_count": int(anchored_encoding.span_match.token_count),
                    "neutral_suffix_token_count": int(neutral_encoding.span_match.token_count),
                    "delta_summary": delta_summary,
                    "figure_path": str(figure_path),
                    "figure_relpath": Path(
                        Path("figures") / "qwen_anchor_carryover_probe" / figure_name
                    ).as_posix(),
                }
            )

        profile_payloads.append(
            {
                "profile": profile_name,
                "groups": sorted({case.anchor_group for case in carryover_cases}),
                "concept_vector_norms": [{"layer": int(layer), **concept_norms[int(layer)]} for layer in layers],
                "cases": case_payloads,
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
