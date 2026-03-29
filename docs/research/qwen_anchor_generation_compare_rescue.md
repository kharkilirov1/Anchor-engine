# Qwen Anchor-Biased Generation Compare

Date: 2026-03-29 16:45 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max new tokens: `12`
Conflict threshold: `0.55`
Bias scale: `1.50`

## Summary

- Cases: `4`
- Base mean lexical consistency score: `1.0000`
- Anchor mean lexical consistency score: `1.0000`
- Anchor minus base mean score: `0.0000`
- Anchor better cases: `none`
- Anchor worse cases: `none`

## Case table

| Family | Case | Mode | Base score | Anchor score | Δ |
|---|---|---|---:|---:|---:|
| quantifier | quantifier_stable | stable | 1.00 | 1.00 | +0.00 |
| quantifier | quantifier_conflict | conflict | 0.00 | 0.00 | +0.00 |
| api_framework | api_framework_stable | stable | 1.00 | 1.00 | +0.00 |
| api_framework | api_framework_conflict | conflict | 2.00 | 2.00 | +0.00 |

## Generated continuations

### quantifier_stable
- base: `The statement "for all natural numbers \( n \) greater`
- anchor: `The statement "for all natural numbers \( n \) greater`
- anchor bias active steps: `3`

### quantifier_conflict
- base: `The statement is true for n = 1, 2`
- anchor: `The statement is true for n = 1, 2`
- anchor bias active steps: `0`

### api_framework_stable
- base: `The FastAPI service is a web application that uses asynchronous request`
- anchor: `The FastAPI service is a web application that uses asynchronous request`
- anchor bias active steps: `1`

### api_framework_conflict
- base: `The text is about a FastAPI service with async request handlers`
- anchor: `The text is about a FastAPI service with async request handlers`
- anchor bias active steps: `1`

## Interpretation

- Это уже не rerank двух вручную заданных продолжений, а реальная greedy generation с token-level anchor bias.
- Метрика здесь грубая и лексическая, поэтому её нельзя считать финальным доказательством. Но она позволяет быстро проверить, толкает ли вмешательство модель в ожидаемую semantic сторону.
