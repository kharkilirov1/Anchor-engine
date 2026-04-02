from __future__ import annotations

from src.utils.anchor_geometry import build_tail_reference_layers, select_tail_probe_layers


def test_select_tail_probe_layers_uses_last_window() -> None:
    assert select_tail_probe_layers(28, count=10) == list(range(18, 28))
    assert select_tail_probe_layers(32, count=10) == list(range(22, 32))


def test_build_tail_reference_layers_tracks_relative_tail_positions() -> None:
    probe_layers = list(range(22, 32))
    reference = build_tail_reference_layers(probe_layers)

    assert reference == {
        "slope_start_layer": 22,
        "slope_end_layer": 28,
        "mature_layer": 28,
        "template_prev_layer": 30,
        "template_curr_layer": 31,
    }
