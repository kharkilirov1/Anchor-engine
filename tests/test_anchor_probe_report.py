from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_anchor_probe_report import generate_report


def test_anchor_probe_report_generation(tmp_path: Path):
    out = tmp_path / 'anchor_probe_report.md'
    path = generate_report(out)
    text = path.read_text(encoding='utf-8')

    assert path.exists()
    assert '# Anchor Probe Report' in text
    assert '## Family deltas' in text
    assert 'forall_stable' in text
    assert 'induction_stable' in text
    assert 'Proposal infl.' in text
    assert 'Blend ratio' in text
    assert 'Strong proposal retire' in text
    assert 'Strong retire gap' in text
    assert '## Current timing bottlenecks' in text
