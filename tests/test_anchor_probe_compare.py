from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_anchor_probe_reports import compare_reports


def test_anchor_probe_report_compare_generation(tmp_path: Path):
    old_report = tmp_path / 'old.md'
    new_report = tmp_path / 'new.md'
    old_report.write_text(
        """# Anchor Probe Report

## Family deltas

| Family | Stable case | Conflict case | Pressure delta | Viability delta | Dead-end delta | Influence delta | Blend delta | Strong proposal retire delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| quantifier | a | b | 0.1000 | 0.2000 | 1 | 0 | 0.0100 | 0 |

## Case table

| Case | Failure mode | Active | Pressure | Viability | Desc. mass | Desc. coherence | Proposal infl. | Proposal score | Blend ratio | Proposal revise | Proposal retire | Strong proposal retire | Strong retire gap | Dead-end | Revisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a | stable | 1 | 0.1000 | 0.9000 | 0.2000 | 0.3000 | 0 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0000 | 0 | 2 |
| b | conflict | 1 | 0.2000 | 0.7000 | 0.4000 | 0.5000 | 1 | 0.6000 | 0.0500 | 1 | 1 | 1 | 0.0400 | 1 | 2 |
""",
        encoding='utf-8',
    )
    new_report.write_text(
        """# Anchor Probe Report

## Family deltas

| Family | Stable case | Conflict case | Pressure delta | Viability delta | Dead-end delta | Influence delta | Blend delta | Strong proposal retire delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| quantifier | a | b | 0.1500 | 0.2500 | 2 | 1 | 0.0300 | 1 |

## Case table

| Case | Failure mode | Active | Pressure | Viability | Desc. mass | Desc. coherence | Proposal infl. | Proposal score | Blend ratio | Proposal revise | Proposal retire | Strong proposal retire | Strong retire gap | Dead-end | Revisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a | stable | 1 | 0.1200 | 0.9500 | 0.2500 | 0.3500 | 0 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0000 | 0 | 2 |
| b | conflict | 1 | 0.2600 | 0.6500 | 0.5000 | 0.4500 | 2 | 0.8000 | 0.1200 | 2 | 1 | 0 | 0.0000 | 2 | 3 |
""",
        encoding='utf-8',
    )
    out = tmp_path / 'compare.md'
    path = compare_reports(old_report, new_report, out)
    text = path.read_text(encoding='utf-8')

    assert path.exists()
    assert '# Anchor Probe Report Comparison' in text
    assert '| quantifier | 0.0500 | 0.0500 | 1.0000 | 1.0000 | 0.0200 | 1.0000 |' in text
    assert '| a | 0.0200 | 0.0500 | 0.0500 | 0.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |' in text
