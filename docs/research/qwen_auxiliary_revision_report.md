# Qwen Auxiliary Revision Report

Date: 2026-03-29 10:07 UTC

## Summary

- Families analyzed: `8`
- Conflict matched-anchor wins: `3/8`
- Conflict revise-gain wins: `4/8`
- Mean match gap (conflict - stable): `0.1250`
- Mean revise-gain gap (conflict - stable): `0.3750`
- Mean retire-delta gap (conflict - stable): `-0.3750`
- Future-rescue families: `2`
- Future-rescue revise-gain wins: `1/2`

## Family table

| Family | Class | Stable matches | Conflict matches | Match gap | Stable revise gain | Conflict revise gain | Revise gap | Stable alt prob | Conflict alt prob |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| api_framework | future_rescue | 1 | 4 | +3 | +0 | +2 | +2 | 0.0811 | 0.2787 |
| entity_property | aligned | 2 | 3 | +1 | +2 | +1 | -1 | 0.4674 | 0.3344 |
| induction | both_weak | 3 | 2 | -1 | +1 | +1 | +0 | 0.3026 | 0.1898 |
| instruction_constraints | delta_only | 3 | 4 | +1 | +1 | +3 | +2 | 0.1326 | 0.1985 |
| legal_scope | delta_only | 2 | 2 | +0 | +2 | +0 | -2 | 0.1686 | 0.0746 |
| proof_mode | aligned | 4 | 4 | +0 | +2 | +3 | +1 | 0.1120 | 0.1636 |
| quantifier | future_rescue | 4 | 2 | -2 | +2 | +1 | -1 | 0.2290 | 0.0711 |
| units | both_weak | 4 | 3 | -1 | +1 | +3 | +2 | 0.3229 | 0.3672 |

## Future-rescue highlights

### api_framework
- match gap: `+3`
- revise-gain gap: `+2`
- stable auxiliary spans: ` the technical`
- conflict auxiliary spans: ` a synchronous Django view;  the text starts`

### quantifier
- match gap: `-2`
- revise-gain gap: `-1`
- stable auxiliary spans: ` a universal mathematical`
- conflict auxiliary spans: ` a witness`

## Interpretation

- This report asks whether auxiliary future-hint proposals change revision behaviour, not just whether they exist.
- A positive revise-gain on conflict cases would mean the proposal-like hints are starting to push the controller toward `revise` rather than the base path.
- If rescue families show stronger matches but little revise gain, the next bottleneck is probably arbiter calibration rather than hint extraction.
