# Qwen Anchor Threshold Calibration

Date: 2026-03-29 03:22 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Seed: `7`
Max length: `192`
Sweeps evaluated: `240`

## Best configuration

- anchor_threshold: `0.10`
- anchor_revision_threshold: `0.35`
- anchor_contradiction_threshold: `0.20`
- anchor_dead_end_threshold: `0.50`
- score: `35.6488`
- pressure wins: `5`
- viability wins: `5`
- joint wins: `5`
- pressure gap: `0.0389`
- viability gap: `0.0260`

## Top configurations

| Rank | anchor | revise | contradiction | dead_end | score | pressure wins | viability wins | joint wins | pressure gap | viability gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.10 | 0.35 | 0.20 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 2 | 0.10 | 0.35 | 0.25 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 3 | 0.10 | 0.35 | 0.30 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 4 | 0.10 | 0.35 | 0.35 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 5 | 0.10 | 0.45 | 0.20 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 6 | 0.10 | 0.45 | 0.25 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 7 | 0.10 | 0.45 | 0.30 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 8 | 0.10 | 0.45 | 0.35 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 9 | 0.10 | 0.55 | 0.20 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |
| 10 | 0.10 | 0.55 | 0.25 | 0.50 | 35.6488 | 5 | 5 | 5 | 0.0389 | 0.0260 |

## Best family breakdown

| Family | Pressure gap | Viability gap | Pressure win | Viability win | Joint win |
|---|---:|---:|---|---|---|
| quantifier | -0.0755 | -0.1890 | False | False | False |
| proof_mode | 0.0884 | 0.0383 | True | True | True |
| induction | 0.0759 | 0.1896 | True | True | True |
| api_framework | -0.0044 | -0.0038 | False | False | False |
| instruction_constraints | 0.1290 | 0.1314 | True | True | True |
| entity_property | 0.1004 | 0.0476 | True | True | True |
| legal_scope | 0.0330 | 0.0882 | True | True | True |
| units | -0.0357 | -0.0943 | False | False | False |

## Interpretation

- Calibration currently optimizes stable-vs-conflict separation on the fixed prompt suite.
- This is still a probe-time heuristic, not a learned objective.
- Proposal counts remain zero in the current overlay, so threshold tuning only improves detector/viability behavior.
