from __future__ import annotations

from pathlib import Path
import re
import sys


METRIC_COLUMNS = {
    'Pressure': 'pressure',
    'Viability': 'viability',
    'Desc. mass': 'desc_mass',
    'Desc. coherence': 'desc_coherence',
    'Dead-end': 'dead_end',
    'Revisions': 'revisions',
}


def _parse_markdown_table(lines: list[str], header_prefix: str) -> tuple[list[str], list[list[str]]]:
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == header_prefix:
            start = idx
            break
    if start is None:
        raise ValueError(f'table header not found: {header_prefix}')

    header = [cell.strip() for cell in lines[start].strip().strip('|').split('|')]
    rows: list[list[str]] = []
    for line in lines[start + 2:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith('|'):
            break
        rows.append([cell.strip() for cell in stripped.strip('|').split('|')])
    return header, rows


def parse_anchor_probe_report(path: Path) -> dict:
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()

    try:
        _, family_rows = _parse_markdown_table(lines, '| Family | Stable case | Conflict case | Pressure delta | Viability delta | Dead-end delta | Influence delta | Blend delta | Strong proposal retire delta |')
        family_format = 'latest'
    except ValueError:
        try:
            _, family_rows = _parse_markdown_table(lines, '| Family | Stable case | Conflict case | Pressure delta | Viability delta | Dead-end delta | Influence delta | Blend delta |')
            family_format = 'proposal'
        except ValueError:
            _, family_rows = _parse_markdown_table(lines, '| Family | Stable case | Conflict case | Pressure delta | Viability delta | Dead-end delta |')
            family_format = 'legacy'

    try:
        _, case_rows = _parse_markdown_table(lines, '| Case | Failure mode | Active | Pressure | Viability | Desc. mass | Desc. coherence | Proposal infl. | Proposal score | Blend ratio | Proposal revise | Proposal retire | Strong proposal retire | Strong retire gap | Dead-end | Revisions |')
        case_format = 'latest'
    except ValueError:
        try:
            _, case_rows = _parse_markdown_table(lines, '| Case | Failure mode | Active | Pressure | Viability | Desc. mass | Desc. coherence | Proposal infl. | Proposal score | Blend ratio | Dead-end | Revisions |')
            case_format = 'proposal'
        except ValueError:
            _, case_rows = _parse_markdown_table(lines, '| Case | Failure mode | Active | Pressure | Viability | Desc. mass | Desc. coherence | Dead-end | Revisions |')
            case_format = 'legacy'

    families = {}
    for row in family_rows:
        parsed = {
            'stable_case': row[1],
            'conflict_case': row[2],
            'pressure_delta': float(row[3]),
            'viability_delta': float(row[4]),
            'dead_end_delta': float(row[5]),
            'influence_delta': 0.0,
            'blend_delta': 0.0,
            'strong_proposal_retire_delta': 0.0,
        }
        if family_format in {'proposal', 'latest'}:
            parsed['influence_delta'] = float(row[6])
            parsed['blend_delta'] = float(row[7])
        if family_format == 'latest':
            parsed['strong_proposal_retire_delta'] = float(row[8])
        families[row[0]] = parsed

    cases = {}
    for row in case_rows:
        parsed = {
            'failure_mode': row[1],
            'active': float(row[2]),
            'pressure': float(row[3]),
            'viability': float(row[4]),
            'desc_mass': float(row[5]),
            'desc_coherence': float(row[6]),
            'proposal_influence': 0.0,
            'proposal_score': 0.0,
            'blend_ratio': 0.0,
            'proposal_revise': 0.0,
            'proposal_retire': 0.0,
            'strong_proposal_retire': 0.0,
            'strong_retire_gap': 0.0,
            'dead_end': 0.0,
            'revisions': 0.0,
        }
        if case_format == 'latest':
            parsed['proposal_influence'] = float(row[7])
            parsed['proposal_score'] = float(row[8])
            parsed['blend_ratio'] = float(row[9])
            parsed['proposal_revise'] = float(row[10])
            parsed['proposal_retire'] = float(row[11])
            parsed['strong_proposal_retire'] = float(row[12])
            parsed['strong_retire_gap'] = float(row[13])
            parsed['dead_end'] = float(row[14])
            parsed['revisions'] = float(row[15])
        elif case_format == 'proposal':
            parsed['proposal_influence'] = float(row[7])
            parsed['proposal_score'] = float(row[8])
            parsed['blend_ratio'] = float(row[9])
            parsed['dead_end'] = float(row[10])
            parsed['revisions'] = float(row[11])
        else:
            parsed['dead_end'] = float(row[7])
            parsed['revisions'] = float(row[8])
        cases[row[0]] = parsed

    return {'families': families, 'cases': cases}


def compare_reports(old_path: Path, new_path: Path, output_path: Path | None = None) -> Path:
    old = parse_anchor_probe_report(old_path)
    new = parse_anchor_probe_report(new_path)

    if output_path is None:
        output_path = new_path.parent / 'anchor_probe_report_compare.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append('# Anchor Probe Report Comparison')
    lines.append('')
    lines.append(f'Old: `{old_path}`')
    lines.append(f'New: `{new_path}`')
    lines.append('')
    lines.append('## Family delta changes')
    lines.append('')
    lines.append('| Family | Δ Pressure delta | Δ Viability delta | Δ Dead-end delta | Δ Influence delta | Δ Blend delta | Δ Strong proposal retire delta |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for family in sorted(set(old['families']) & set(new['families'])):
        old_row = old['families'][family]
        new_row = new['families'][family]
        lines.append(
            f"| {family} | {new_row['pressure_delta'] - old_row['pressure_delta']:.4f} | {new_row['viability_delta'] - old_row['viability_delta']:.4f} | {new_row['dead_end_delta'] - old_row['dead_end_delta']:.4f} | {new_row['influence_delta'] - old_row['influence_delta']:.4f} | {new_row['blend_delta'] - old_row['blend_delta']:.4f} | {new_row['strong_proposal_retire_delta'] - old_row['strong_proposal_retire_delta']:.4f} |"
        )
    lines.append('')
    lines.append('## Case metric changes')
    lines.append('')
    lines.append('| Case | Δ Pressure | Δ Viability | Δ Desc. mass | Δ Desc. coherence | Δ Proposal infl. | Δ Proposal score | Δ Blend ratio | Δ Proposal revise | Δ Proposal retire | Δ Strong proposal retire | Δ Strong retire gap | Δ Dead-end |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for case in sorted(set(old['cases']) & set(new['cases'])):
        old_row = old['cases'][case]
        new_row = new['cases'][case]
        lines.append(
            f"| {case} | {new_row['pressure'] - old_row['pressure']:.4f} | {new_row['viability'] - old_row['viability']:.4f} | {new_row['desc_mass'] - old_row['desc_mass']:.4f} | {new_row['desc_coherence'] - old_row['desc_coherence']:.4f} | {new_row['proposal_influence'] - old_row['proposal_influence']:.4f} | {new_row['proposal_score'] - old_row['proposal_score']:.4f} | {new_row['blend_ratio'] - old_row['blend_ratio']:.4f} | {new_row['proposal_revise'] - old_row['proposal_revise']:.4f} | {new_row['proposal_retire'] - old_row['proposal_retire']:.4f} | {new_row['strong_proposal_retire'] - old_row['strong_proposal_retire']:.4f} | {new_row['strong_retire_gap'] - old_row['strong_retire_gap']:.4f} | {new_row['dead_end'] - old_row['dead_end']:.4f} |"
        )
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- Positive family pressure delta means the stable-vs-conflict gap widened in the new report.')
    lines.append('- Positive family viability delta means the stable-vs-conflict viability gap widened in the new report.')
    lines.append('- For case rows, values are `new - old`. Interpreting direction depends on whether the case is stable or conflict.')
    lines.append('')

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python scripts/compare_anchor_probe_reports.py <old_report.md> <new_report.md> [output.md]')
    old_path = Path(sys.argv[1]).resolve()
    new_path = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve() if len(sys.argv) >= 4 else None
    path = compare_reports(old_path, new_path, output)
    print(path)
