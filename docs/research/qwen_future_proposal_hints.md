# Qwen Future Proposal Hints

Date: 2026-03-29 07:39 UTC

## Summary

- Hint candidates: `11`
- Source families: `4`

## Candidate spans

| Family | Case | Class | Hint score | Span | Mean | Max | Text |
|---|---|---|---:|---|---:|---:|---|
| api_framework | api_framework_conflict | future_rescue | 1.6242 | 19-22 | 0.8828 | 1.0000 | ` a synchronous Django view` |
| api_framework | api_framework_conflict | future_rescue | 1.5236 | 26-26 | 0.8281 | 0.8281 | ` the` |
| api_framework | api_framework_conflict | future_rescue | 1.5092 | 13-15 | 0.8203 | 0.8594 | ` the text starts` |
| quantifier | quantifier_conflict | future_rescue | 1.0723 | 24-25 | 0.9531 | 1.0000 | ` a witness` |
| quantifier | quantifier_conflict | future_rescue | 0.8570 | 29-29 | 0.7617 | 0.7617 | ` the` |
| entity_property | entity_property_conflict | aligned | 1.9102 | 25-25 | 1.0000 | 1.0000 | `-line` |
| entity_property | entity_property_conflict | aligned | 1.7833 | 22-23 | 0.9336 | 0.9727 | ` the routine` |
| entity_property | entity_property_conflict | aligned | 1.7833 | 29-30 | 0.9336 | 0.9727 | ` the clinical` |
| proof_mode | proof_mode_conflict | aligned | 1.4336 | 21-21 | 1.0000 | 1.0000 | ` proof` |
| proof_mode | proof_mode_conflict | aligned | 1.2824 | 10-10 | 0.8945 | 0.8945 | ` Half` |
| proof_mode | proof_mode_conflict | aligned | 1.2656 | 19-19 | 0.8828 | 0.8828 | ` a` |

## Interpretation

- These are conflict-case future-influence spans that do not overlap the current active anchors.
- `future_rescue` families are the most interesting, because they are cases where future-conditioned attribution may expose missed anchor spans.
- Current future-rescue families: `api_framework, quantifier`
- The next practical use for these spans is as experimental proposal hints or auxiliary anchor candidates, not as automatic replacements for the current detector.
