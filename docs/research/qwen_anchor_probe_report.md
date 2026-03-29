# Qwen Anchor Probe Report

Date: 2026-03-29 03:09 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max length: `192`
Seed: `7`

## Summary

- Cases: `16`
- Stable cases: `8`
- Conflict cases: `8`
- Conflict minus stable pressure gap: `0.0389`
- Conflict minus stable viability gap: `-0.0167`

## Case table

| Family | Case | Expected | Tokens | Active | Pressure | Viability | Dead ends | Proposals |
|---|---|---|---:|---:|---:|---:|---:|---:|
| quantifier | quantifier_stable | stable | 27 | 3 | 0.6039 | 0.3285 | 3 | 0 |
| quantifier | quantifier_conflict | conflict | 32 | 5 | 0.5284 | 0.5175 | 1 | 0 |
| proof_mode | proof_mode_stable | stable | 31 | 2 | 0.6364 | 0.2358 | 4 | 0 |
| proof_mode | proof_mode_conflict | conflict | 27 | 2 | 0.7248 | 0.1975 | 4 | 0 |
| induction | induction_stable | stable | 29 | 3 | 0.6961 | 0.2868 | 3 | 0 |
| induction | induction_conflict | conflict | 34 | 1 | 0.7721 | 0.0972 | 5 | 0 |
| api_framework | api_framework_stable | stable | 30 | 1 | 0.7795 | 0.0906 | 5 | 0 |
| api_framework | api_framework_conflict | conflict | 30 | 1 | 0.7752 | 0.0945 | 5 | 0 |
| instruction_constraints | instruction_constraints_stable | stable | 28 | 2 | 0.6190 | 0.2359 | 4 | 0 |
| instruction_constraints | instruction_constraints_conflict | conflict | 36 | 2 | 0.7479 | 0.1842 | 4 | 0 |
| entity_property | entity_property_stable | stable | 32 | 1 | 0.6735 | 0.1405 | 5 | 0 |
| entity_property | entity_property_conflict | conflict | 33 | 1 | 0.7739 | 0.0929 | 5 | 0 |
| legal_scope | legal_scope_stable | stable | 28 | 3 | 0.6079 | 0.3241 | 3 | 0 |
| legal_scope | legal_scope_conflict | conflict | 37 | 2 | 0.6409 | 0.2359 | 4 | 0 |
| units | units_stable | stable | 31 | 1 | 0.7559 | 0.0950 | 5 | 0 |
| units | units_conflict | conflict | 46 | 2 | 0.7202 | 0.1838 | 4 | 0 |

## Interpretation

- This report is diagnostic only; it does not yet apply proposal-guided decoding.
- Conflict-tagged prompts show higher contradiction pressure than stable prompts in this run (`+0.0389`), which is the direction we want.
- Conflict-tagged prompts show lower viability than stable prompts in this run (`-0.0167`), which is also the intended direction.
- Proposal counts remaining at zero indicate that proposal-path activation is still unresolved in the current overlay.
