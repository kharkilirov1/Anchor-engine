# Qwen Anchor Probe Report

Date: 2026-03-29 03:03 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max length: `192`
Seed: `7`

## Summary

- Cases: `5`
- Stable cases: `2`
- Conflict cases: `3`
- Conflict minus stable pressure gap: `0.0261`
- Conflict minus stable viability gap: `-0.0378`

## Case table

| Case | Expected | Tokens | Active | Pressure | Viability | Dead ends | Proposals |
|---|---|---:|---:|---:|---:|---:|---:|
| quantifier_stable | stable | 27 | 3 | 0.6039 | 0.3285 | 3 | 0 |
| quantifier_conflict | conflict | 32 | 5 | 0.5284 | 0.5175 | 1 | 0 |
| proof_mode_conflict | conflict | 27 | 2 | 0.7248 | 0.1975 | 4 | 0 |
| induction_stable | stable | 29 | 3 | 0.6961 | 0.2868 | 3 | 0 |
| api_framework_conflict | conflict | 30 | 1 | 0.7752 | 0.0945 | 5 | 0 |

## Interpretation

- This report is diagnostic only; it does not yet apply proposal-guided decoding.
- Conflict-tagged prompts show higher contradiction pressure than stable prompts in this run (`+0.0261`), which is the direction we want.
- Conflict-tagged prompts show lower viability than stable prompts in this run (`-0.0378`), which is also the intended direction.
- Proposal counts remaining at zero indicate that proposal-path activation is still unresolved in the current overlay.
