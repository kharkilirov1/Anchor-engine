# Qwen Auxiliary Revision Report

Date: 2026-03-29 11:16 UTC

## Summary

- Families analyzed: `8`
- Conflict matched-anchor wins: `3/8`
- Conflict revise-gain wins: `3/8`
- Mean match gap (conflict - stable): `0.2500`
- Mean revise-gain gap (conflict - stable): `0.2500`
- Mean retire-delta gap (conflict - stable): `-0.1250`
- Future-rescue families: `2`
- Future-rescue revise-gain wins: `1/2`

## Family table

| Family | Class | Stable matches | Conflict matches | Match gap | Stable revise gain | Conflict revise gain | Revise gap | Stable alt prob | Conflict alt prob |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| api_framework | future_rescue | 1 | 2 | +1 | +0 | +2 | +2 | 0.0811 | 0.4701 |
| entity_property | aligned | 1 | 2 | +1 | +1 | +1 | +0 | 0.7023 | 0.4636 |
| induction | both_weak | 1 | 2 | +1 | +1 | +1 | +0 | 0.6983 | 0.1898 |
| instruction_constraints | delta_only | 2 | 2 | +0 | +1 | +1 | +0 | 0.1724 | 0.3063 |
| legal_scope | delta_only | 2 | 1 | -1 | +2 | +0 | -2 | 0.1686 | 0.0809 |
| proof_mode | aligned | 2 | 2 | +0 | +1 | +2 | +1 | 0.1564 | 0.2371 |
| quantifier | future_rescue | 1 | 1 | +0 | +1 | +1 | +0 | 0.6366 | 0.0823 |
| units | both_weak | 2 | 2 | +0 | +1 | +2 | +1 | 0.5674 | 0.4632 |

## Future-rescue highlights

### api_framework
- match gap: `+1`
- revise-gain gap: `+2`
- stable auxiliary spans: ` the technical`
- conflict auxiliary spans: ` a synchronous Django view;  the text starts`

### quantifier
- match gap: `+0`
- revise-gain gap: `+0`
- stable auxiliary spans: ` a universal mathematical`
- conflict auxiliary spans: ` a witness`

## Interpretation

- This report asks whether auxiliary future-hint proposals change revision behaviour, not just whether they exist.
- A positive revise-gain on conflict cases would mean the proposal-like hints are starting to push the controller toward `revise` rather than the base path.
- If rescue families show stronger matches but little revise gain, the next bottleneck is probably arbiter calibration rather than hint extraction.
