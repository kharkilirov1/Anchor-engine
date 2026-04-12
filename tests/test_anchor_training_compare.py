import json
from pathlib import Path

from scripts.generate_anchor_training_compare import generate_compare_report


def test_generate_anchor_training_compare_report(tmp_path):
    baseline = [
        {"step": 0.0, "loss": 5.0, "val_bpb": 4.0},
        {"step": 10.0, "loss": 3.0, "val_bpb": 2.5},
    ]
    anchor = [
        {
            "step": 0.0,
            "loss": 5.2,
            "val_bpb": 4.2,
            "anchors_active": 2.0,
            "anchor_contradiction": 0.8,
            "anchor_viability": 0.2,
            "proposal_influence": 0.0,
        },
        {
            "step": 10.0,
            "loss": 2.8,
            "val_bpb": 2.3,
            "anchors_active": 5.0,
            "anchor_contradiction": 0.5,
            "anchor_viability": 0.4,
            "proposal_influence": 1.0,
        },
    ]
    baseline_path = tmp_path / "baseline.json"
    anchor_path = tmp_path / "anchor.json"
    output_path = tmp_path / "compare.md"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    anchor_path.write_text(json.dumps(anchor), encoding="utf-8")

    generated = generate_compare_report(baseline_path, anchor_path, output_path)
    text = generated.read_text(encoding="utf-8")

    assert generated == output_path
    assert "# Anchor vs Baseline Training Compare" in text
    assert "| val_bpb | 2.5000 | 2.3000 | -0.2000 |" in text
    assert "Proposal path became active" in text
