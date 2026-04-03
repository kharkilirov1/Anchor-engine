"""
ABPT Geometry 3D Visualizer
============================
Читает данные из ABPT archive и строит интерактивные 3D визуализации:
  1. Траектории concept vectors по слоям (PCA в 3D)
  2. Pairwise cosine heatmap между группами по слоям
  3. Concept vector norms по слоям
  4. Carryover delta по слоям и группам

Использование:
  python visualize_abpt_geometry_3d.py
  python visualize_abpt_geometry_3d.py --profile medium
  python visualize_abpt_geometry_3d.py --watch  # авто-обновление при новых данных
"""
from __future__ import annotations

import argparse
import json
import time
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
OUTPUT_DIR = ROOT / "docs" / "research" / "figures" / "3d_geometry"

DIRECTION_MAP_JSON = ARCHIVE / "qwen35_4b_anchor_concept_direction_map.json"
CARRYOVER_JSON = ARCHIVE / "qwen35_4b_anchor_carryover_probe.json"

GROUP_COLORS = {
    "strictly_vegan_meal_plan_policy":              "#2ecc71",
    "async_fastapi_service_architecture_policy":    "#3498db",
    "json_only_response_format_policy":             "#e74c3c",
    "proof_by_contradiction_reasoning_steps":       "#9b59b6",
    "binary_search_update_loop_procedure":          "#f39c12",
    "dependency_injection_request_flow_sequence":   "#1abc9c",
}

GROUP_SHORT = {
    "strictly_vegan_meal_plan_policy":              "vegan",
    "async_fastapi_service_architecture_policy":    "fastapi",
    "json_only_response_format_policy":             "json",
    "proof_by_contradiction_reasoning_steps":       "contradiction",
    "binary_search_update_loop_procedure":          "binary",
    "dependency_injection_request_flow_sequence":   "DI",
}

CARRYOVER_LAYERS = {
    "strictly_vegan_meal_plan_policy":              11,
    "async_fastapi_service_architecture_policy":    None,
    "json_only_response_format_policy":             11,
    "proof_by_contradiction_reasoning_steps":       25,
    "binary_search_update_loop_procedure":          10,
    "dependency_injection_request_flow_sequence":   24,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_direction_map(profile: str = "medium") -> dict:
    data = json.loads(DIRECTION_MAP_JSON.read_text())
    for p in data["profiles"]:
        if p["profile"] == profile:
            return p
    raise ValueError(f"Profile '{profile}' not found")


def load_carryover(profile: str = "medium") -> dict:
    data = json.loads(CARRYOVER_JSON.read_text())
    for p in data["profiles"]:
        if p["profile"] == profile:
            return p
    raise ValueError(f"Profile '{profile}' not found")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Concept vector norms по слоям
# ─────────────────────────────────────────────────────────────────────────────

def chart_concept_norms(profile_data: dict, profile: str) -> go.Figure:
    norms = profile_data["concept_vector_norms"]
    layers = [row["layer"] for row in norms]
    groups = [g for g in norms[0] if g != "layer"]

    fig = go.Figure()
    for g in groups:
        vals = [row[g] for row in norms]
        color = GROUP_COLORS.get(g, "#999")
        short = GROUP_SHORT.get(g, g[:10])
        carry_layer = CARRYOVER_LAYERS.get(g)

        fig.add_trace(go.Scatter(
            x=layers, y=vals,
            name=short,
            line=dict(color=color, width=2),
            mode="lines+markers",
            marker=dict(size=4),
            hovertemplate=f"<b>{short}</b><br>Layer: %{{x}}<br>Norm: %{{y:.3f}}<extra></extra>",
        ))

        # Маркер carryover слоя
        if carry_layer is not None and carry_layer < len(layers):
            carry_val = norms[carry_layer][g]
            fig.add_trace(go.Scatter(
                x=[carry_layer], y=[carry_val],
                mode="markers",
                marker=dict(symbol="star", size=12, color=color,
                            line=dict(color="white", width=1)),
                name=f"{short} carryover@L{carry_layer}",
                showlegend=False,
                hovertemplate=f"<b>{short} carryover</b><br>Layer: {carry_layer}<br>Norm: {carry_val:.3f}<extra></extra>",
            ))

    # Зона кристаллизации L4-L8
    fig.add_vrect(x0=4, x1=8, fillcolor="rgba(255,255,0,0.08)",
                  line_width=0, annotation_text="Birth L4-L8",
                  annotation_position="top left")

    fig.update_layout(
        title=f"Concept Vector Norms по слоям — профиль: {profile}",
        xaxis_title="Layer",
        yaxis_title="Norm",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: PCA 3D траектории concept vectors по слоям
# ─────────────────────────────────────────────────────────────────────────────

def chart_pca_trajectories(profile_data: dict, profile: str) -> go.Figure:
    from sklearn.decomposition import PCA

    norms = profile_data["concept_vector_norms"]
    groups = [g for g in norms[0] if g != "layer"]
    layers = [row["layer"] for row in norms]
    n_layers = len(layers)

    # Матрица: для каждой группы берём norm по слоям как 1D feature
    # + pairwise cosines для обогащения
    pairwise = profile_data["concept_pairwise_cosines"]

    # Строим feature matrix: каждая точка = (group, layer)
    # features = [norm, cos_to_others...]
    points = []
    labels_group = []
    labels_layer = []

    for g in groups:
        for i, layer in enumerate(layers):
            norm_val = norms[i][g]
            # косинусы с другими группами на этом слое
            layer_key = str(layer)
            cos_features = []
            if layer_key in pairwise:
                for g2 in groups:
                    if g2 != g:
                        cos_val = pairwise[layer_key].get(g, {}).get(g2, 0.0)
                        cos_features.append(cos_val)
            feature_vec = [norm_val] + cos_features
            points.append(feature_vec)
            labels_group.append(g)
            labels_layer.append(layer)

    X = np.array(points, dtype=np.float32)
    # Нормализуем
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    pca = PCA(n_components=3)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    fig = go.Figure()

    for g in groups:
        color = GROUP_COLORS.get(g, "#999")
        short = GROUP_SHORT.get(g, g[:10])
        mask = [i for i, lg in enumerate(labels_group) if lg == g]
        x = coords[mask, 0]
        y = coords[mask, 1]
        z = coords[mask, 2]
        layer_vals = [labels_layer[i] for i in mask]

        # Линия траектории
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines+markers",
            name=short,
            line=dict(color=color, width=3),
            marker=dict(
                size=[6 if l not in (4, 5, 6, 7, 8) else 10 for l in layer_vals],
                color=layer_vals,
                colorscale="Viridis",
                opacity=0.9,
                line=dict(color=color, width=1),
            ),
            hovertext=[f"L{l}" for l in layer_vals],
            hovertemplate=f"<b>{short}</b> %{{hovertext}}<br>PC1=%{{x:.2f}} PC2=%{{y:.2f}} PC3=%{{z:.2f}}<extra></extra>",
        ))

        # Маркер точки кристаллизации (L4-L8 peak)
        birth_mask = [i for i, (idx, l) in enumerate(zip(mask, layer_vals)) if 4 <= l <= 8]
        if birth_mask:
            peak_i = birth_mask[0]  # L4
            fig.add_trace(go.Scatter3d(
                x=[x[peak_i]], y=[y[peak_i]], z=[z[peak_i]],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color=color,
                            line=dict(color="white", width=2)),
                name=f"{short} birth",
                showlegend=False,
                hovertemplate=f"<b>{short} birth zone</b><extra></extra>",
            ))

    var_str = " | ".join([f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(explained)])
    fig.update_layout(
        title=f"3D PCA: Concept Direction Trajectories — {profile}<br><sub>Explained variance: {var_str}</sub>",
        scene=dict(
            xaxis_title=f"PC1 ({explained[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({explained[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({explained[2]*100:.1f}%)",
            bgcolor="rgb(10,10,20)",
        ),
        template="plotly_dark",
        height=650,
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Pairwise cosine heatmap по слоям
# ─────────────────────────────────────────────────────────────────────────────

def chart_pairwise_heatmap(profile_data: dict, profile: str) -> go.Figure:
    pairwise = profile_data["concept_pairwise_cosines"]
    groups = [g for g in profile_data["groups"]]
    short_names = [GROUP_SHORT.get(g, g[:8]) for g in groups]
    layers = sorted([int(k) for k in pairwise.keys()])

    # Для каждого слоя строим матрицу n_groups × n_groups
    # Показываем mean off-diagonal cosine по слоям как линию
    mean_off_diag = []
    for layer in layers:
        lk = str(layer)
        vals = []
        for g1 in groups:
            for g2 in groups:
                if g1 != g2:
                    v = pairwise[lk].get(g1, {}).get(g2, 0.0)
                    vals.append(v)
        mean_off_diag.append(np.mean(vals) if vals else 0.0)

    # Heatmap финального слоя (L31)
    final_layer = str(max(layers))
    matrix = []
    for g1 in groups:
        row = []
        for g2 in groups:
            v = pairwise[final_layer].get(g1, {}).get(g2, 0.0)
            row.append(v)
        matrix.append(row)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Pairwise cosine heatmap @ L{max(layers)}",
            "Mean off-diagonal cosine по слоям",
        ],
        column_widths=[0.55, 0.45],
    )

    fig.add_trace(go.Heatmap(
        z=matrix,
        x=short_names,
        y=short_names,
        colorscale="RdBu",
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>cosine: %{z:.3f}<extra></extra>",
        showscale=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=layers,
        y=mean_off_diag,
        mode="lines+markers",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=5),
        name="mean off-diag",
        hovertemplate="L%{x}<br>mean cosine: %{y:.3f}<extra></extra>",
    ), row=1, col=2)

    fig.add_vrect(x0=4, x1=8, fillcolor="rgba(255,255,0,0.08)",
                  line_width=0, row=1, col=2)

    fig.update_layout(
        title=f"Group Separation — профиль: {profile}",
        template="plotly_dark",
        height=450,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Carryover delta по группам и слоям
# ─────────────────────────────────────────────────────────────────────────────

def chart_carryover(carryover_data: dict, profile: str) -> go.Figure:
    cases = carryover_data["cases"]
    group_names = list({c["anchor_group"] for c in cases})
    n_layers = 32

    fig = go.Figure()

    for g in group_names:
        case = next((c for c in cases if c["anchor_group"] == g), None)
        if case is None:
            continue
        ds = case["delta_summary"]
        layer_means = ds.get("layer_mean_deltas", [])
        if not layer_means:
            continue

        layers = [row["layer"] for row in layer_means]
        deltas = [row["mean_delta"] for row in layer_means]
        color = GROUP_COLORS.get(g, "#999")
        short = GROUP_SHORT.get(g, g[:10])
        peak_layer = ds.get("peak_delta_layer")
        peak_val = ds.get("peak_delta_value", 0)

        fig.add_trace(go.Scatter(
            x=layers, y=deltas,
            name=short,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{short}</b><br>L%{{x}}: %{{y:.4f}}<extra></extra>",
        ))

        if peak_layer is not None:
            fig.add_trace(go.Scatter(
                x=[peak_layer], y=[peak_val],
                mode="markers",
                marker=dict(symbol="star", size=14, color=color,
                            line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate=f"<b>{short} peak</b><br>L{peak_layer}: {peak_val:.3f}<extra></extra>",
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_vrect(x0=4, x1=8, fillcolor="rgba(255,255,0,0.08)", line_width=0,
                  annotation_text="Birth", annotation_position="top left")

    fig.update_layout(
        title=f"Carryover Delta по слоям — профиль: {profile}<br><sub>★ = peak carryover layer per group</sub>",
        xaxis_title="Layer",
        yaxis_title="Δ cosine (anchored − neutral)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5: Сводный профиль — все метрики в одном
# ─────────────────────────────────────────────────────────────────────────────

def chart_summary_phase_profile(profile_data: dict, carryover_data: dict, profile: str) -> go.Figure:
    """Показывает birth / propagation / integration / handoff фазы наглядно."""
    norms = profile_data["concept_vector_norms"]
    layers = [row["layer"] for row in norms]
    groups = [g for g in norms[0] if g != "layer"]

    # Среднее norm по всем группам
    mean_norms = []
    for row in norms:
        vals = [row[g] for g in groups]
        mean_norms.append(np.mean(vals))

    # Mean carryover delta
    cases = carryover_data["cases"]
    layer_carryover_mean = {}
    for case in cases:
        for lrow in case["delta_summary"].get("layer_mean_deltas", []):
            l = lrow["layer"]
            layer_carryover_mean[l] = layer_carryover_mean.get(l, []) + [lrow["mean_delta"]]
    carryover_layers = sorted(layer_carryover_mean.keys())
    carryover_means = [np.mean(layer_carryover_mean[l]) for l in carryover_layers]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Mean Concept Vector Norm", "Mean Carryover Delta"],
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=layers, y=mean_norms,
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        name="mean norm",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=carryover_layers, y=carryover_means,
        mode="lines+markers",
        line=dict(color="#e74c3c", width=2),
        name="mean carryover Δ",
    ), row=2, col=1)

    # Фазовые зоны
    phases = [
        (0, 3, "rgba(100,100,100,0.15)", "Pre-semantic"),
        (4, 8, "rgba(255,255,0,0.10)", "Birth"),
        (9, 15, "rgba(0,255,150,0.07)", "Propagation"),
        (16, 23, "rgba(0,150,255,0.07)", "Integration"),
        (24, 31, "rgba(255,100,0,0.07)", "Handoff"),
    ]
    for x0, x1, color, label in phases:
        for row in [1, 2]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0,
                          annotation_text=label if row == 1 else "",
                          annotation_position="top left",
                          row=row, col=1)

    fig.update_layout(
        title=f"Phase Profile — профиль: {profile}",
        template="plotly_dark",
        height=600,
        xaxis2_title="Layer",
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML Dashboard builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(profile: str = "medium") -> Path:
    print(f"[ABPT Visualizer] Загружаю данные (профиль: {profile})...")

    dir_data = load_direction_map(profile)
    carry_data = load_carryover(profile)

    print("[ABPT Visualizer] Строю графики...")

    charts = [
        chart_summary_phase_profile(dir_data, carry_data, profile),
        chart_pca_trajectories(dir_data, profile),
        chart_concept_norms(dir_data, profile),
        chart_carryover(carry_data, profile),
        chart_pairwise_heatmap(dir_data, profile),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"abpt_geometry_dashboard_{profile}.html"

    # Собираем HTML
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ABPT Geometry Dashboard</title>
<style>
  body { background: #0d0d1a; color: #eee; font-family: monospace; margin: 0; padding: 20px; }
  h1 { color: #3498db; border-bottom: 1px solid #333; padding-bottom: 10px; }
  .meta { color: #888; font-size: 12px; margin-bottom: 20px; }
  .chart { margin-bottom: 30px; border: 1px solid #222; border-radius: 6px; overflow: hidden; }
  .phase-legend { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
  .phase { padding: 4px 12px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<h1>🔬 ABPT Geometry Dashboard</h1>
<div class="meta">
  Model: Qwen/Qwen3.5-4B &nbsp;|&nbsp; Profile: """ + profile + """ &nbsp;|&nbsp;
  <span style="color:#ff0">★</span> = peak carryover layer &nbsp;|&nbsp;
  <span style="color:#ff0">◆</span> = birth zone marker
</div>
<div class="phase-legend">
  <div class="phase" style="background:rgba(100,100,100,0.3)">L0-3: Pre-semantic</div>
  <div class="phase" style="background:rgba(255,255,0,0.2)">L4-8: Birth / Crystallization</div>
  <div class="phase" style="background:rgba(0,255,150,0.15)">L9-15: Propagation</div>
  <div class="phase" style="background:rgba(0,150,255,0.15)">L16-23: Context Integration</div>
  <div class="phase" style="background:rgba(255,100,0,0.2)">L24-31: Generation Handoff</div>
</div>
"""]

    for fig in charts:
        html_parts.append('<div class="chart">')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if len(html_parts) == 2 else False))
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    output_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"[ABPT Visualizer] ✅ Сохранено: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Bridge: авто-обновление при изменении архива
# ─────────────────────────────────────────────────────────────────────────────

def watch_and_rebuild(profile: str = "medium", interval: int = 10) -> None:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class ArchiveHandler(FileSystemEventHandler):
        def __init__(self):
            self.last_rebuild = 0

        def on_modified(self, event):
            if event.src_path.endswith(".json"):
                now = time.time()
                if now - self.last_rebuild > 5:  # debounce
                    self.last_rebuild = now
                    print(f"\n[Bridge] Изменение: {event.src_path}")
                    try:
                        path = build_dashboard(profile)
                        print(f"[Bridge] Dashboard обновлён → {path}")
                    except Exception as e:
                        print(f"[Bridge] Ошибка: {e}")

    handler = ArchiveHandler()
    observer = Observer()
    observer.schedule(handler, str(ARCHIVE), recursive=False)
    observer.start()
    print(f"[Bridge] Слежу за {ARCHIVE} (профиль: {profile})")
    print("[Bridge] Ctrl+C для остановки")
    try:
        while True:
            time.sleep(interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ABPT Geometry 3D Visualizer")
    parser.add_argument("--profile", default="medium",
                        choices=["short", "medium", "long"],
                        help="Anchor length profile")
    parser.add_argument("--watch", action="store_true",
                        help="Авто-обновление при изменении архива")
    parser.add_argument("--no-open", action="store_true",
                        help="Не открывать браузер")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Построить dashboard для всех трёх профилей")
    args = parser.parse_args()

    if args.all_profiles:
        paths = []
        for p in ["short", "medium", "long"]:
            paths.append(build_dashboard(p))
        print(f"\n[ABPT Visualizer] Построены все профили:")
        for path in paths:
            print(f"  {path}")
        if not args.no_open:
            webbrowser.open(str(paths[1]))  # открываем medium
    else:
        path = build_dashboard(args.profile)
        if not args.no_open:
            webbrowser.open(str(path))

    if args.watch:
        watch_and_rebuild(args.profile)


if __name__ == "__main__":
    main()
