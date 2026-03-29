# Qwen Span Miss Analysis

Date: 2026-03-29 06:55 UTC

## Summary

- Families analyzed: `8`
- Fully aligned families: `2`
- Future-rescue families: `2`
- Delta-only families: `2`
- Both-weak families: `2`

## Family table

| Family | Pressure gap | Viability gap | Anchor-future gap | Span-overlap gap | Classification |
|---|---:|---:|---:|---:|---|
| api_framework | -0.0044 | 0.0038 | 0.8398 | 0.2500 | future_rescue |
| entity_property | 0.1004 | -0.0476 | 0.9102 | 0.2500 | aligned |
| induction | 0.0759 | -0.1896 | 0.5378 | -0.0833 | both_weak |
| instruction_constraints | 0.1290 | -0.1314 | -0.3210 | 0.0000 | delta_only |
| legal_scope | 0.0330 | -0.0882 | -0.3216 | -0.2500 | delta_only |
| proof_mode | 0.0884 | -0.0383 | 0.4336 | 0.2500 | aligned |
| quantifier | -0.0755 | 0.1890 | 0.1251 | 0.0000 | future_rescue |
| units | -0.0357 | 0.0943 | -0.1159 | -0.0833 | both_weak |

## Interpretation

- `aligned` means delta diagnostics and future-attribution spans both move in the expected conflict direction.
- `future_rescue` means the current detector misses the family, but anchor-position future influence still rises on the conflict case.
- `delta_only` means the existing detector works better than the future-attribution overlay on that family.
- `both_weak` means neither signal is currently convincing enough.

## Current reading

- The most interesting families are `future_rescue`, because they are candidates where future-conditioned attribution may expose missed anchor spans.
- The most important near-term target is `both_weak`, because those families likely need better prompts, better thresholds, or a stronger span aggregation rule.
