# Qwen Auxiliary Proposal Report

Date: 2026-03-29 09:07 UTC

## Summary

- Families analyzed: `8`
- Conflict proposal-count wins: `5/8`
- Conflict proposal-score wins: `5/8`
- Joint wins (count + score): `2/8`
- Mean proposal-count gap (conflict - stable): `0.6250`
- Mean proposal-score gap (conflict - stable): `0.0352`
- Future-rescue families: `2`
- Future-rescue joint wins: `0/2`

## Family table

| Family | Class | Stable count | Conflict count | Count gap | Stable score | Conflict score | Score gap |
|---|---|---:|---:|---:|---:|---:|---:|
| api_framework | future_rescue | 1 | 2 | +1 | 0.8906 | 0.8516 | -0.0391 |
| entity_property | aligned | 2 | 3 | +1 | 0.8906 | 0.9557 | +0.0651 |
| induction | both_weak | 1 | 3 | +2 | 0.9609 | 0.8607 | -0.1003 |
| instruction_constraints | delta_only | 2 | 3 | +1 | 0.8926 | 0.8620 | -0.0306 |
| legal_scope | delta_only | 2 | 1 | -1 | 0.8691 | 1.0000 | +0.1309 |
| proof_mode | aligned | 3 | 3 | +0 | 0.8581 | 0.9076 | +0.0495 |
| quantifier | future_rescue | 1 | 1 | +0 | 0.8750 | 0.9531 | +0.0781 |
| units | both_weak | 2 | 3 | +1 | 0.8281 | 0.9557 | +0.1276 |

## Future-rescue highlights

### api_framework
- count gap: `+1`
- score gap: `-0.0391`
- stable auxiliary spans: ` the technical`
- conflict auxiliary spans: ` a synchronous Django view;  the text starts`

### quantifier
- count gap: `+0`
- score gap: `+0.0781`
- stable auxiliary spans: ` a universal mathematical`
- conflict auxiliary spans: ` a witness`

## Interpretation

- This report evaluates the new auxiliary proposal-like spans built from high future-influence regions.
- A useful result is not absolute proposal volume, but whether conflict prompts produce more or stronger auxiliary proposals than their stable controls.
- `future_rescue` families are the most important target, because they are where future-conditioned attribution may compensate for detector misses.
- These auxiliary proposals are still offline diagnostics; they are not yet connected to decoding or revision control.
