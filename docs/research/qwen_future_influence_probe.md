# Qwen Future Influence Probe

Date: 2026-03-29 04:34 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max length: `192`
Future window: `16`
Seed: `7`

## Summary

- Cases: `16`
- Stable cases: `8`
- Conflict cases: `8`
- Conflict minus stable mean future influence gap: `-0.0250`
- Conflict minus stable active-anchor future influence gap: `0.2610`

## Case table

| Family | Case | Expected | Tokens | Active | Mean future influence | Anchor-position mean | Max influence | Future loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| quantifier | quantifier_stable | stable | 27 | 3 | 0.3867 | 0.3617 | 1.0000 | 4.2500 |
| quantifier | quantifier_conflict | conflict | 32 | 5 | 0.3555 | 0.4867 | 1.0000 | 5.2188 |
| proof_mode | proof_mode_stable | stable | 31 | 2 | 0.2637 | 0.0000 | 1.0000 | 3.2031 |
| proof_mode | proof_mode_conflict | conflict | 27 | 2 | 0.3984 | 0.4336 | 1.0000 | 3.6562 |
| induction | induction_stable | stable | 29 | 3 | 0.3105 | 0.2669 | 1.0000 | 3.1875 |
| induction | induction_conflict | conflict | 34 | 1 | 0.2988 | 0.8047 | 1.0000 | 3.9062 |
| api_framework | api_framework_stable | stable | 30 | 1 | 0.3145 | 0.0000 | 1.0000 | 3.2656 |
| api_framework | api_framework_conflict | conflict | 30 | 1 | 0.3730 | 0.8398 | 1.0000 | 5.1875 |
| instruction_constraints | instruction_constraints_stable | stable | 28 | 3 | 0.4355 | 0.5221 | 1.0000 | 4.4062 |
| instruction_constraints | instruction_constraints_conflict | conflict | 36 | 2 | 0.3477 | 0.2012 | 1.0000 | 5.7812 |
| entity_property | entity_property_stable | stable | 32 | 1 | 0.2969 | 0.0000 | 1.0000 | 3.7188 |
| entity_property | entity_property_conflict | conflict | 33 | 1 | 0.3281 | 0.9102 | 1.0000 | 3.5938 |
| legal_scope | legal_scope_stable | stable | 28 | 3 | 0.4316 | 0.3216 | 1.0000 | 4.3750 |
| legal_scope | legal_scope_conflict | conflict | 37 | 2 | 0.2295 | 0.0000 | 1.0000 | 4.9062 |
| units | units_stable | stable | 31 | 2 | 0.3125 | 0.2109 | 1.0000 | 3.9219 |
| units | units_conflict | conflict | 46 | 3 | 0.2207 | 0.0951 | 1.0000 | 3.6094 |

## Top future-influence tokens

### quantifier_stable
- pos `18` | token ` a` | id `264` | score `1.0000`
- pos `23` | token ` universal` | id `20178` | score `0.8984`
- pos `11` | token ` the` | id `279` | score `0.8867`
- pos `15` | token `.` | id `13` | score `0.8789`
- pos `24` | token ` mathematical` | id `35972` | score `0.8789`

### quantifier_conflict
- pos `24` | token ` a` | id `264` | score `1.0000`
- pos `18` | token ` text` | id `1467` | score `0.9062`
- pos `25` | token ` witness` | id `11298` | score `0.9062`
- pos `19` | token ` shifts` | id `28635` | score `0.8672`
- pos `27` | token `.` | id `13` | score `0.8281`

### proof_mode_stable
- pos `19` | token ` a` | id `264` | score `1.0000`
- pos `28` | token `iction` | id `2479` | score `0.8633`
- pos `22` | token ` preserves` | id `74898` | score `0.8594`
- pos `14` | token ` claim` | id `3717` | score `0.8516`
- pos `16` | token `.` | id `13` | score `0.7695`

### proof_mode_conflict
- pos `21` | token ` proof` | id `11064` | score `1.0000`
- pos `10` | token ` Half` | id `25839` | score `0.8945`
- pos `19` | token ` a` | id `264` | score `0.8828`
- pos `23` | token `.` | id `13` | score `0.8672`
- pos `16` | token ` starts` | id `8471` | score `0.8281`

### induction_stable
- pos `26` | token ` induction` | id `37056` | score `1.0000`
- pos `25` | token ` in` | id `304` | score `0.9219`
- pos `21` | token `.` | id `13` | score `0.8008`
- pos `18` | token ` the` | id `279` | score `0.7578`
- pos `16` | token `,` | id `11` | score `0.7422`

### induction_conflict
- pos `26` | token ` the` | id `279` | score `1.0000`
- pos `27` | token ` induction` | id `37056` | score `0.9922`
- pos `23` | token ` example` | id `3110` | score `0.8789`
- pos `29` | token `.` | id `13` | score `0.8047`
- pos `22` | token ` arbitrary` | id `24168` | score `0.7930`

### api_framework_stable
- pos `20` | token `.` | id `13` | score `1.0000`
- pos `15` | token ` and` | id `323` | score `0.9766`
- pos `26` | token ` that` | id `429` | score `0.9336`
- pos `27` | token ` same` | id `1852` | score `0.9336`
- pos `22` | token ` the` | id `279` | score `0.9297`

### api_framework_conflict
- pos `22` | token ` view` | id `1651` | score `1.0000`
- pos `20` | token ` synchronous` | id `65949` | score `0.8672`
- pos `13` | token ` the` | id `279` | score `0.8594`
- pos `19` | token ` a` | id `264` | score `0.8398`
- pos `24` | token `.` | id `13` | score `0.8398`

### instruction_constraints_stable
- pos `25` | token ` those` | id `1846` | score `1.0000`
- pos `15` | token ` avoid` | id `5648` | score `0.9648`
- pos `21` | token ` the` | id `279` | score `0.9570`
- pos `17` | token `.` | id `13` | score `0.9492`
- pos `19` | token ` by` | id `553` | score `0.9336`

### instruction_constraints_conflict
- pos `24` | token ` casual` | id `16334` | score `1.0000`
- pos `21` | token ` starts` | id `8471` | score `0.9336`
- pos `33` | token ` the` | id `279` | score `0.9180`
- pos `29` | token ` creative` | id `11521` | score `0.8867`
- pos `25` | token ` long` | id `1293` | score `0.8711`

### entity_property_stable
- pos `29` | token ` allergy` | id `59654` | score `1.0000`
- pos `28` | token ` that` | id `429` | score `0.9258`
- pos `21` | token `.` | id `13` | score `0.8984`
- pos `24` | token ` clinical` | id `14490` | score `0.8789`
- pos `23` | token ` the` | id `279` | score `0.8516`

### entity_property_conflict
- pos `25` | token `-line` | id `8447` | score `1.0000`
- pos `22` | token ` the` | id `279` | score `0.9727`
- pos `30` | token ` clinical` | id `14490` | score `0.9727`
- pos `16` | token ` starts` | id `8471` | score `0.9375`
- pos `27` | token `.` | id `13` | score `0.9102`

### legal_scope_stable
- pos `24` | token ` those` | id `1846` | score `1.0000`
- pos `18` | token `.` | id `13` | score `0.9648`
- pos `16` | token `-commercial` | id `73044` | score `0.9336`
- pos `15` | token ` non` | id `2477` | score `0.8555`
- pos `23` | token ` preserving` | id `46895` | score `0.8555`

### legal_scope_conflict
- pos `29` | token ` commercial` | id `8353` | score `1.0000`
- pos `24` | token ` claim` | id `3717` | score `0.7188`
- pos `20` | token ` the` | id `279` | score `0.7070`
- pos `23` | token ` to` | id `311` | score `0.7031`
- pos `27` | token ` for` | id `369` | score `0.7031`

### units_stable
- pos `22` | token ` calculation` | id `21937` | score `1.0000`
- pos `19` | token `.` | id `13` | score `0.8984`
- pos `28` | token ` unit` | id `4982` | score `0.8828`
- pos `26` | token ` the` | id `279` | score `0.8203`
- pos `23` | token ` notes` | id `8388` | score `0.8125`

### units_conflict
- pos `43` | token ` calculation` | id `21937` | score `1.0000`
- pos `32` | token ` were` | id `1033` | score `0.9922`
- pos `36` | token ` meters` | id `20044` | score `0.9648`
- pos `33` | token ` already` | id `2669` | score `0.9258`
- pos `40` | token `.` | id `13` | score `0.8984`

## Interpretation

- This report is an experimental midpoint between delta-hidden heuristics and full leave-one-out KL.
- Scores are based on gradient influence of token positions on a future autoregressive loss window.
- High-scoring positions are candidates for semantically important context even when local hidden-state jumps are ambiguous.
