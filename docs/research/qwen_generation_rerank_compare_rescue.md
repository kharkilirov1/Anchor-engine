# Qwen Base vs Anchor Rerank Compare

Date: 2026-03-29 16:01 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max length: `192`
Future window: `16`
Rerank strength: `0.35`

## Summary

- Cases: `4`
- Base accuracy: `0.2500`
- Anchor rerank accuracy: `0.2500`
- Accuracy delta (anchor - base): `+0` cases
- Stable base accuracy: `0.5000`
- Stable anchor accuracy: `0.5000`
- Conflict base accuracy: `0.0000`
- Conflict anchor accuracy: `0.0000`

- Rescued cases: `none`
- Regressed cases: `none`

## Case table

| Family | Case | Mode | Base ok | Anchor ok | Base margin | Anchor margin | Δ vs base |
|---|---|---|---|---|---:|---:|---:|
| quantifier | quantifier_stable | stable | no | no | -0.1406 | -0.1755 | -0.0349 |
| quantifier | quantifier_conflict | conflict | no | no | -1.2031 | -1.2109 | -0.0077 |
| api_framework | api_framework_stable | stable | yes | yes | 1.1562 | 1.2597 | 0.1035 |
| api_framework | api_framework_conflict | conflict | no | no | -0.3125 | -0.3504 | -0.0379 |

## Candidate diagnostics

### quantifier_stable
- preferred base logprob `-4.0000`, anchor bonus `-0.1307`, rerank `-4.0457`
- rejected base logprob `-3.8594`, anchor bonus `-0.0311`, rerank `-3.8703`
- preferred pressure `0.5668` vs rejected `0.6429`
- preferred viability `0.4216` vs rejected `0.2306`
- preferred revise gain `+1` vs rejected `+3`
- preferred continuation: `Therefore the continuation keeps the universal claim and does not switch to an existential witness.`
- rejected continuation: `Therefore it is enough to give one witness, so the universal claim can be dropped.`

### quantifier_conflict
- preferred base logprob `-4.4688`, anchor bonus `-0.0852`, rerank `-4.4986`
- rejected base logprob `-3.2656`, anchor bonus `-0.0631`, rerank `-3.2877`
- preferred pressure `0.5763` vs rejected `0.6121`
- preferred viability `0.4150` vs rejected `0.3197`
- preferred revise gain `+1` vs rejected `+2`
- preferred continuation: `The correction is to reject that existential drift and restore the original universal statement.`
- rejected continuation: `So the proof now only needs one witness and the universal statement no longer matters.`

### api_framework_stable
- preferred base logprob `-2.4844`, anchor bonus `-0.6927`, rerank `-2.7268`
- rejected base logprob `-3.6406`, anchor bonus `-0.9883`, rerank `-3.9865`
- preferred pressure `0.6451` vs rejected `0.8092`
- preferred viability `0.2310` vs rejected `0.0000`
- preferred revise gain `+0` vs rejected `+1`
- preferred continuation: `Keep the explanation on async FastAPI handlers, dependency injection, and Pydantic validation.`
- rejected continuation: `Reframe the service as a synchronous Django class-based view with template rendering.`

### api_framework_conflict
- preferred base logprob `-3.9844`, anchor bonus `-0.8003`, rerank `-4.2645`
- rejected base logprob `-3.6719`, anchor bonus `-0.6920`, rerank `-3.9141`
- preferred pressure `0.7792` vs rejected `0.6441`
- preferred viability `0.0901` vs rejected `0.2308`
- preferred revise gain `+1` vs rejected `+0`
- preferred continuation: `Correct the drift and return to async FastAPI handlers with typed request models.`
- rejected continuation: `Continue describing a synchronous Django view and middleware stack.`

## Interpretation

- Это не свободная генерация, а constrained reranking между двумя короткими продолжениями на один prompt.
- Такой тест слабее настоящего decoding benchmark, но уже позволяет увидеть, даёт ли anchor-side сигнал хоть какой-то полезный приоритет поверх base model.
- Если anchor rerank выигрывает хотя бы некоторые cases, следующий шаг — перенести ту же логику в мягкий logits bias или beam rerank на реальной генерации.
