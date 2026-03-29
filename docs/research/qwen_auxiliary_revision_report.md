# Qwen Auxiliary Revision Report

Date: 2026-03-29 13:52 UTC

## Summary

- Families analyzed: `8`
- Conflict matched-anchor wins: `3/8`
- Conflict revise-gain wins: `4/8`
- Mean match gap (conflict - stable): `0.3750`
- Mean revise-gain gap (conflict - stable): `0.5000`
- Mean retire-delta gap (conflict - stable): `-0.3750`
- Future-rescue families: `2`
- Future-rescue revise-gain wins: `1/2`

## Family table

| Family | Class | Stable matches | Conflict matches | Match gap | Stable revise gain | Conflict revise gain | Revise gap | Stable alt prob | Conflict alt prob |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| api_framework | future_rescue | 0 | 2 | +2 | +0 | +2 | +2 | 0.0000 | 0.4567 |
| entity_property | aligned | 1 | 2 | +1 | +1 | +1 | +0 | 0.5989 | 0.4695 |
| induction | both_weak | 1 | 2 | +1 | +1 | +2 | +1 | 0.5983 | 0.2964 |
| instruction_constraints | delta_only | 2 | 2 | +0 | +1 | +2 | +1 | 0.2906 | 0.3610 |
| legal_scope | delta_only | 2 | 1 | -1 | +2 | +1 | -1 | 0.2759 | 0.2075 |
| proof_mode | aligned | 2 | 2 | +0 | +2 | +2 | +0 | 0.2634 | 0.3496 |
| quantifier | future_rescue | 1 | 1 | +0 | +1 | +1 | +0 | 0.5442 | 0.2084 |
| units | both_weak | 2 | 2 | +0 | +1 | +2 | +1 | 0.5019 | 0.4883 |

## Future-rescue highlights

### api_framework
- match gap: `+2`
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
