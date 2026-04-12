import json
from pathlib import Path

from scripts.generate_anchor_training_report import generate_training_report


def test_generate_anchor_training_report(tmp_path: Path):
    history_path = tmp_path / "history.json"
    report_path = tmp_path / "anchor_training_report.md"
    history = [
        {
            "step": 0.0,
            "loss": 4.5,
            "ce_loss": 4.4,
            "bpb": 6.3,
            "val_bpb": 6.2,
            "anchors_active": 120.0,
            "anchor_contradiction": 0.72,
            "anchor_viability": 0.31,
            "anchor_dead_end": 66.0,
            "proposal_influence": 0.0,
            "proposal_blend": 0.0,
            "strong_retire_gap": 0.18,
            "detector_alignment_loss": 0.11,
            "context_stability_loss": 0.07,
        },
        {
            "step": 10.0,
            "loss": 4.1,
            "ce_loss": 4.0,
            "bpb": 5.9,
            "val_bpb": 5.8,
            "anchors_active": 128.0,
            "anchor_contradiction": 0.61,
            "anchor_viability": 0.41,
            "anchor_dead_end": 50.0,
            "proposal_influence": 2.0,
            "proposal_blend": 0.08,
            "strong_retire_gap": 0.12,
            "detector_alignment_loss": 0.08,
            "context_stability_loss": 0.05,
        },
    ]
    history_path.write_text(json.dumps(history), encoding="utf-8")

    path = generate_training_report(history_path, report_path)

    assert path == report_path
    text = report_path.read_text(encoding="utf-8")
    assert "# Anchor Training Report" in text
    assert "| loss | 4.5000 | 4.1000 | -0.4000 | 4.1000 |" in text
    assert "| anchor_contradiction | 0.7200 | 0.6100 | -0.1100 | 0.6100 |" in text
    assert "| 10.0000 | 4.1000 | 4.0000 | 5.9000 | 5.8000 |" in text
