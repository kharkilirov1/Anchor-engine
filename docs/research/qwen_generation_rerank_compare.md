# Qwen Base vs Anchor Rerank Compare

Date: 2026-03-29 16:22 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max length: `192`
Future window: `16`
Rerank strength: `0.35`

## Summary

- Cases: `16`
- Base accuracy: `0.4375`
- Anchor rerank accuracy: `0.5000`
- Accuracy delta (anchor - base): `+1` cases
- Stable base accuracy: `0.6250`
- Stable anchor accuracy: `0.7500`
- Conflict base accuracy: `0.2500`
- Conflict anchor accuracy: `0.2500`

- Rescued cases: `proof_mode_stable`
- Regressed cases: `none`

## Case table

| Family | Case | Mode | Base ok | Anchor ok | Base margin | Anchor margin | Δ vs base |
|---|---|---|---|---|---:|---:|---:|
| quantifier | quantifier_stable | stable | no | no | -0.1406 | -0.1755 | -0.0349 |
| quantifier | quantifier_conflict | conflict | no | no | -1.2031 | -1.2109 | -0.0077 |
| proof_mode | proof_mode_stable | stable | no | yes | -0.0625 | 0.0137 | 0.0762 |
| proof_mode | proof_mode_conflict | conflict | yes | yes | 0.7188 | 0.7190 | 0.0003 |
| induction | induction_stable | stable | yes | yes | 0.3750 | 0.2980 | -0.0770 |
| induction | induction_conflict | conflict | no | no | -0.0469 | -0.1463 | -0.0994 |
| api_framework | api_framework_stable | stable | yes | yes | 1.1562 | 1.2597 | 0.1035 |
| api_framework | api_framework_conflict | conflict | no | no | -0.3125 | -0.3504 | -0.0379 |
| instruction_constraints | instruction_constraints_stable | stable | yes | yes | 2.7188 | 2.6557 | -0.0630 |
| instruction_constraints | instruction_constraints_conflict | conflict | yes | yes | 0.7344 | 0.6573 | -0.0770 |
| entity_property | entity_property_stable | stable | yes | yes | 0.0938 | 0.0544 | -0.0393 |
| entity_property | entity_property_conflict | conflict | no | no | -0.4219 | -0.4881 | -0.0662 |
| legal_scope | legal_scope_stable | stable | yes | yes | 2.0625 | 2.0982 | 0.0357 |
| legal_scope | legal_scope_conflict | conflict | no | no | -0.3125 | -0.3343 | -0.0218 |
| units | units_stable | stable | no | no | -0.2344 | -0.1515 | 0.0828 |
| units | units_conflict | conflict | no | no | -1.7656 | -1.7605 | 0.0051 |

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

### proof_mode_stable
- preferred base logprob `-4.1562`, anchor bonus `-0.2407`, rerank `-4.2405`
- rejected base logprob `-4.0938`, anchor bonus `-0.4586`, rerank `-4.2543`
- preferred pressure `0.6369` vs rejected `0.6345`
- preferred viability `0.2360` vs rejected `0.2361`
- preferred revise gain `+2` vs rejected `+1`
- preferred continuation: `We therefore keep the contradiction structure and discharge the assumed negation.`
- rejected continuation: `We now abandon contradiction mode and switch to a direct constructive proof from scratch.`

### proof_mode_conflict
- preferred base logprob `-3.3125`, anchor bonus `-0.4297`, rerank `-3.4629`
- rejected base logprob `-4.0312`, anchor bonus `-0.4305`, rerank `-4.1819`
- preferred pressure `0.6413` vs rejected `0.6421`
- preferred viability `0.2305` vs rejected `0.2304`
- preferred revise gain `+1` vs rejected `+1`
- preferred continuation: `The continuation should return to contradiction mode and conclude that the assumption was false.`
- rejected continuation: `The continuation should drop contradiction and proceed as an ordinary direct proof.`

### induction_stable
- preferred base logprob `-2.7500`, anchor bonus `-0.5893`, rerank `-2.9563`
- rejected base logprob `-3.1250`, anchor bonus `-0.3693`, rerank `-3.2542`
- preferred pressure `0.7339` vs rejected `0.7339`
- preferred viability `0.1903` vs rejected `0.1902`
- preferred revise gain `+1` vs rejected `+2`
- preferred continuation: `The next move is to apply the induction hypothesis in the step from n to n plus one.`
- rejected continuation: `A single illustrative example is enough, so the induction step is unnecessary.`

### induction_conflict
- preferred base logprob `-3.1719`, anchor bonus `-0.6016`, rerank `-3.3824`
- rejected base logprob `-3.1250`, anchor bonus `-0.3176`, rerank `-3.2362`
- preferred pressure `0.6923` vs rejected `0.5853`
- preferred viability `0.2905` vs rejected `0.4102`
- preferred revise gain `+0` vs rejected `+0`
- preferred continuation: `Repair the argument by reinstating the induction hypothesis and proving the successor case.`
- rejected continuation: `Continue from the arbitrary example and ignore the induction hypothesis entirely.`

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

### instruction_constraints_stable
- preferred base logprob `-2.7188`, anchor bonus `-0.2535`, rerank `-2.8075`
- rejected base logprob `-5.4375`, anchor bonus `-0.0735`, rerank `-5.4632`
- preferred pressure `0.6194` vs rejected `0.6192`
- preferred viability `0.3155` vs rejected `0.3153`
- preferred revise gain `+1` vs rejected `+2`
- preferred continuation: `Use exactly three formal bullet points and avoid speculative language.`
- rejected continuation: `Write a casual paragraph with creative guesses and no bullet structure.`

### instruction_constraints_conflict
- preferred base logprob `-3.6406`, anchor bonus `-0.2564`, rerank `-3.7304`
- rejected base logprob `-4.3750`, anchor bonus `-0.0362`, rerank `-4.3877`
- preferred pressure `0.5947` vs rejected `0.5946`
- preferred viability `0.3321` vs rejected `0.3322`
- preferred revise gain `+1` vs rejected `+2`
- preferred continuation: `Return to three formal bullet points and avoid unsupported guesses.`
- rejected continuation: `Lean into a casual long narrative with speculative commentary.`

### entity_property_stable
- preferred base logprob `-3.2969`, anchor bonus `-0.6611`, rerank `-3.5282`
- rejected base logprob `-3.3906`, anchor bonus `-0.5487`, rerank `-3.5827`
- preferred pressure `0.6750` vs rejected `0.8097`
- preferred viability `0.1405` vs rejected `0.0000`
- preferred revise gain `+1` vs rejected `+3`
- preferred continuation: `Continue with a safe non-penicillin alternative and keep the allergy warning explicit.`
- rejected continuation: `Recommend amoxicillin as the routine next step despite the documented allergy.`

### entity_property_conflict
- preferred base logprob `-3.0000`, anchor bonus `-0.8478`, rerank `-3.2967`
- rejected base logprob `-2.5781`, anchor bonus `-0.6585`, rerank `-2.8086`
- preferred pressure `0.7709` vs rejected `0.6718`
- preferred viability `0.0956` vs rejected `0.1401`
- preferred revise gain `+1` vs rejected `+1`
- preferred continuation: `Flag the contradiction and avoid amoxicillin because of the penicillin allergy.`
- rejected continuation: `Proceed with amoxicillin as the routine recommendation.`

### legal_scope_stable
- preferred base logprob `-2.1875`, anchor bonus `-0.1134`, rerank `-2.2272`
- rejected base logprob `-4.2500`, anchor bonus `-0.2153`, rerank `-4.3254`
- preferred pressure `0.6082` vs rejected `0.7130`
- preferred viability `0.3266` vs rejected `0.2738`
- preferred revise gain `+2` vs rejected `+2`
- preferred continuation: `The clause remains limited to Kazakhstan and to non-commercial use only.`
- rejected continuation: `The clause now grants worldwide commercial sublicensing rights.`

### legal_scope_conflict
- preferred base logprob `-2.9844`, anchor bonus `-0.3409`, rerank `-3.1037`
- rejected base logprob `-2.6719`, anchor bonus `-0.2786`, rerank `-2.7694`
- preferred pressure `0.6132` vs rejected `0.6117`
- preferred viability `0.3232` vs rejected `0.3244`
- preferred revise gain `+1` vs rejected `+1`
- preferred continuation: `Narrow the language back to Kazakhstan-only non-commercial use.`
- rejected continuation: `Expand the rights to worldwide commercial sublicensing.`

### units_stable
- preferred base logprob `-3.0156`, anchor bonus `-0.3673`, rerank `-3.1442`
- rejected base logprob `-2.7812`, anchor bonus `-0.6040`, rerank `-2.9926`
- preferred pressure `0.7319` vs rejected `0.7184`
- preferred viability `0.1904` vs rejected `0.2695`
- preferred revise gain `+2` vs rejected `+0`
- preferred continuation: `Keep the result in centimeters and avoid any unit change without conversion.`
- rejected continuation: `Treat 125 as meters without converting from centimeters.`

### units_conflict
- preferred base logprob `-3.3750`, anchor bonus `-0.3370`, rerank `-3.4930`
- rejected base logprob `-1.6094`, anchor bonus `-0.3516`, rerank `-1.7324`
- preferred pressure `0.6603` vs rejected `0.6699`
- preferred viability `0.3081` vs rejected `0.3015`
- preferred revise gain `+1` vs rejected `+1`
- preferred continuation: `Correct the calculation by converting units before making any statement in meters.`
- rejected continuation: `Continue as if 125 cm already means 125 meters.`

## Interpretation

- Это не свободная генерация, а constrained reranking между двумя короткими продолжениями на один prompt.
- Такой тест слабее настоящего decoding benchmark, но уже позволяет увидеть, даёт ли anchor-side сигнал хоть какой-то полезный приоритет поверх base model.
- Если anchor rerank выигрывает хотя бы некоторые cases, следующий шаг — перенести ту же логику в мягкий logits bias или beam rerank на реальной генерации.
