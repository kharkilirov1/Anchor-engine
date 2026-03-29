# Qwen Future Influence Probe

Date: 2026-03-29 13:50 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max length: `192`
Future window: `16`
Span threshold: `0.75`
Top spans per case: `4`
Seed: `7`

## Summary

- Cases: `16`
- Stable cases: `8`
- Conflict cases: `8`
- Conflict minus stable mean future influence gap: `-0.0250`
- Conflict minus stable active-anchor future influence gap: `0.2610`
- Conflict minus stable future-span overlap gap: `0.0417`
- Conflict minus stable auxiliary proposal-count gap: `0.6250`
- Conflict minus stable auxiliary proposal-score gap: `0.0352`
- Conflict minus stable auxiliary revision-match gap: `0.3750`
- Conflict minus stable auxiliary revise-gain gap: `0.5000`
- Conflict minus stable auxiliary retire-delta gap: `-0.3750`

## Case table

| Family | Case | Expected | Tokens | Active | Aux proposals | Aux matches | Aux revise gain | Mean future influence | Anchor-position mean | Span overlap | Max influence | Future loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| quantifier | quantifier_stable | stable | 27 | 3 | 1 | 1 | +1 | 0.3867 | 0.3617 | 0.5000 | 1.0000 | 4.2500 |
| quantifier | quantifier_conflict | conflict | 32 | 5 | 1 | 1 | +1 | 0.3555 | 0.4867 | 0.5000 | 1.0000 | 5.2188 |
| proof_mode | proof_mode_stable | stable | 31 | 2 | 3 | 2 | +2 | 0.2637 | 0.0000 | 0.0000 | 1.0000 | 3.2031 |
| proof_mode | proof_mode_conflict | conflict | 27 | 2 | 3 | 2 | +2 | 0.3984 | 0.4336 | 0.2500 | 1.0000 | 3.6562 |
| induction | induction_stable | stable | 29 | 3 | 1 | 1 | +1 | 0.3105 | 0.2669 | 0.3333 | 1.0000 | 3.1875 |
| induction | induction_conflict | conflict | 34 | 1 | 3 | 2 | +2 | 0.2988 | 0.8047 | 0.2500 | 1.0000 | 3.9062 |
| api_framework | api_framework_stable | stable | 30 | 1 | 1 | 0 | +0 | 0.3145 | 0.0000 | 0.0000 | 1.0000 | 3.2656 |
| api_framework | api_framework_conflict | conflict | 30 | 1 | 2 | 2 | +2 | 0.3730 | 0.8398 | 0.2500 | 1.0000 | 5.1875 |
| instruction_constraints | instruction_constraints_stable | stable | 28 | 3 | 2 | 2 | +1 | 0.4355 | 0.5221 | 0.2500 | 1.0000 | 4.4062 |
| instruction_constraints | instruction_constraints_conflict | conflict | 36 | 2 | 3 | 2 | +2 | 0.3477 | 0.2012 | 0.2500 | 1.0000 | 5.7812 |
| entity_property | entity_property_stable | stable | 32 | 1 | 2 | 1 | +1 | 0.2969 | 0.0000 | 0.0000 | 1.0000 | 3.7188 |
| entity_property | entity_property_conflict | conflict | 33 | 1 | 3 | 2 | +1 | 0.3281 | 0.9102 | 0.2500 | 1.0000 | 3.5938 |
| legal_scope | legal_scope_stable | stable | 28 | 3 | 2 | 2 | +2 | 0.4316 | 0.3216 | 0.2500 | 1.0000 | 4.3750 |
| legal_scope | legal_scope_conflict | conflict | 37 | 2 | 1 | 1 | +1 | 0.2295 | 0.0000 | 0.0000 | 1.0000 | 4.9062 |
| units | units_stable | stable | 31 | 2 | 2 | 2 | +1 | 0.3125 | 0.2109 | 0.3333 | 1.0000 | 3.9219 |
| units | units_conflict | conflict | 46 | 3 | 3 | 2 | +2 | 0.2207 | 0.0951 | 0.2500 | 1.0000 | 3.6094 |

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

## High future-influence spans

### quantifier_stable
- span `18-18` | mean `1.0000` | max `1.0000` | text ` a`
- span `11-11` | mean `0.8867` | max `0.8867` | text ` the`
- span `15-15` | mean `0.8789` | max `0.8789` | text `.`
- span `22-24` | mean `0.8750` | max `0.8984` | text ` a universal mathematical`
- active anchor spans: `14-15, 18-19, 25-26`
- future-span overlap ratio: `0.5000` | anchor-span overlap ratio: `0.6667`
- proposal-like future hint spans:
  - `22-24` | mean `0.8750` | text ` a universal mathematical`
- auxiliary proposals:
  - `22-24` | score `0.8750` | text ` a universal mathematical`
- auxiliary revision: matches `1`, mean alt prob `0.5442`, revise gain `+1`, retire delta `-1`

### quantifier_conflict
- span `24-25` | mean `0.9531` | max `1.0000` | text ` a witness`
- span `17-20` | mean `0.8516` | max `0.9062` | text ` the text shifts toward`
- span `27-27` | mean `0.8281` | max `0.8281` | text `.`
- span `29-29` | mean `0.7617` | max `0.7617` | text ` the`
- active anchor spans: `11-12, 14-15, 17-18, 26-27, 30-31`
- future-span overlap ratio: `0.5000` | anchor-span overlap ratio: `0.4000`
- proposal-like future hint spans:
  - `24-25` | mean `0.9531` | text ` a witness`
- auxiliary proposals:
  - `24-25` | score `0.9531` | text ` a witness`
- auxiliary revision: matches `1`, mean alt prob `0.2084`, revise gain `+1`, retire delta `+0`

### proof_mode_stable
- span `19-19` | mean `1.0000` | max `1.0000` | text ` a`
- span `28-28` | mean `0.8633` | max `0.8633` | text `iction`
- span `22-22` | mean `0.8594` | max `0.8594` | text ` preserves`
- span `14-14` | mean `0.8516` | max `0.8516` | text ` claim`
- active anchor spans: `7-8, 29-30`
- future-span overlap ratio: `0.0000` | anchor-span overlap ratio: `0.0000`
- proposal-like future hint spans:
  - `28-28` | mean `0.8633` | text `iction`
  - `22-22` | mean `0.8594` | text ` preserves`
  - `14-14` | mean `0.8516` | text ` claim`
- auxiliary proposals:
  - `28-28` | score `0.8633` | text `iction`
  - `22-22` | score `0.8594` | text ` preserves`
  - `14-14` | score `0.8516` | text ` claim`
- auxiliary revision: matches `2`, mean alt prob `0.2634`, revise gain `+2`, retire delta `-2`

### proof_mode_conflict
- span `21-21` | mean `1.0000` | max `1.0000` | text ` proof`
- span `10-10` | mean `0.8945` | max `0.8945` | text ` Half`
- span `19-19` | mean `0.8828` | max `0.8828` | text ` a`
- span `23-24` | mean `0.8438` | max `0.8672` | text `. Continue`
- active anchor spans: `7-8, 22-23`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `0.5000`
- proposal-like future hint spans:
  - `21-21` | mean `1.0000` | text ` proof`
  - `10-10` | mean `0.8945` | text ` Half`
- auxiliary proposals:
  - `21-21` | score `1.0000` | text ` proof`
  - `10-10` | score `0.8945` | text ` Half`
  - `16-16` | score `0.8281` | text ` starts`
- auxiliary revision: matches `2`, mean alt prob `0.3496`, revise gain `+2`, retire delta `-1`

### induction_stable
- span `25-26` | mean `0.9609` | max `1.0000` | text ` in induction`
- span `21-21` | mean `0.8008` | max `0.8008` | text `.`
- span `18-18` | mean `0.7578` | max `0.7578` | text ` the`
- active anchor spans: `2-3, 7-8, 20-21`
- future-span overlap ratio: `0.3333` | anchor-span overlap ratio: `0.3333`
- proposal-like future hint spans:
  - `25-26` | mean `0.9609` | text ` in induction`
- auxiliary proposals:
  - `25-26` | score `0.9609` | text ` in induction`
- auxiliary revision: matches `1`, mean alt prob `0.5983`, revise gain `+1`, retire delta `-1`

### induction_conflict
- span `26-27` | mean `0.9961` | max `1.0000` | text ` the induction`
- span `22-23` | mean `0.8359` | max `0.8789` | text ` arbitrary example`
- span `29-29` | mean `0.8047` | max `0.8047` | text `.`
- span `18-18` | mean `0.7500` | max `0.7500` | text ` starts`
- active anchor spans: `28-29`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `1.0000`
- proposal-like future hint spans:
  - `26-27` | mean `0.9961` | text ` the induction`
  - `22-23` | mean `0.8359` | text ` arbitrary example`
  - `18-18` | mean `0.7500` | text ` starts`
- auxiliary proposals:
  - `26-27` | score `0.9961` | text ` the induction`
  - `22-23` | score `0.8359` | text ` arbitrary example`
  - `18-18` | score `0.7500` | text ` starts`
- auxiliary revision: matches `2`, mean alt prob `0.2964`, revise gain `+2`, retire delta `-2`

### api_framework_stable
- span `20-20` | mean `1.0000` | max `1.0000` | text `.`
- span `15-15` | mean `0.9766` | max `0.9766` | text ` and`
- span `26-27` | mean `0.9336` | max `0.9336` | text ` that same`
- span `22-23` | mean `0.8906` | max `0.9297` | text ` the technical`
- active anchor spans: `10-11`
- future-span overlap ratio: `0.0000` | anchor-span overlap ratio: `0.0000`
- proposal-like future hint spans:
  - `22-23` | mean `0.8906` | text ` the technical`
- auxiliary proposals:
  - `22-23` | score `0.8906` | text ` the technical`
- auxiliary revision: matches `0`, mean alt prob `0.0000`, revise gain `+0`, retire delta `+0`

### api_framework_conflict
- span `19-22` | mean `0.8828` | max `1.0000` | text ` a synchronous Django view`
- span `24-24` | mean `0.8398` | max `0.8398` | text `.`
- span `26-26` | mean `0.8281` | max `0.8281` | text ` the`
- span `13-15` | mean `0.8203` | max `0.8594` | text ` the text starts`
- active anchor spans: `23-24`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `1.0000`
- proposal-like future hint spans:
  - `19-22` | mean `0.8828` | text ` a synchronous Django view`
  - `13-15` | mean `0.8203` | text ` the text starts`
- auxiliary proposals:
  - `19-22` | score `0.8828` | text ` a synchronous Django view`
  - `13-15` | score `0.8203` | text ` the text starts`
- auxiliary revision: matches `2`, mean alt prob `0.4567`, revise gain `+2`, retire delta `-2`

### instruction_constraints_stable
- span `17-17` | mean `0.9492` | max `0.9492` | text `.`
- span `19-19` | mean `0.9336` | max `0.9336` | text ` by`
- span `21-25` | mean `0.9062` | max `1.0000` | text ` the answer while preserving those`
- span `14-15` | mean `0.8789` | max `0.9648` | text ` and avoid`
- active anchor spans: `10-11, 17-18, 26-27`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `0.3333`
- proposal-like future hint spans:
  - `21-25` | mean `0.9062` | text ` the answer while preserving those`
  - `14-15` | mean `0.8789` | text ` and avoid`
- auxiliary proposals:
  - `21-25` | score `0.9062` | text ` the answer while preserving those`
  - `14-15` | score `0.8789` | text ` and avoid`
- auxiliary revision: matches `2`, mean alt prob `0.2906`, revise gain `+1`, retire delta `+0`

### instruction_constraints_conflict
- span `19-21` | mean `0.8633` | max `0.9336` | text ` the text starts`
- span `33-34` | mean `0.8633` | max `0.9180` | text ` the planning`
- span `31-31` | mean `0.8633` | max `0.8633` | text `.`
- span `28-29` | mean `0.8594` | max `0.8867` | text ` with creative`
- active anchor spans: `10-11, 31-32`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `0.5000`
- proposal-like future hint spans:
  - `19-21` | mean `0.8633` | text ` the text starts`
  - `33-34` | mean `0.8633` | text ` the planning`
  - `28-29` | mean `0.8594` | text ` with creative`
- auxiliary proposals:
  - `19-21` | score `0.8633` | text ` the text starts`
  - `33-34` | score `0.8633` | text ` the planning`
  - `28-29` | score `0.8594` | text ` with creative`
- auxiliary revision: matches `2`, mean alt prob `0.3610`, revise gain `+2`, retire delta `-1`

### entity_property_stable
- span `28-29` | mean `0.9609` | max `1.0000` | text ` that allergy`
- span `21-21` | mean `0.8984` | max `0.8984` | text `.`
- span `23-26` | mean `0.8203` | max `0.8789` | text ` the clinical note while`
- active anchor spans: `30-31`
- future-span overlap ratio: `0.0000` | anchor-span overlap ratio: `0.0000`
- proposal-like future hint spans:
  - `28-29` | mean `0.9609` | text ` that allergy`
  - `23-26` | mean `0.8203` | text ` the clinical note while`
- auxiliary proposals:
  - `28-29` | score `0.9609` | text ` that allergy`
  - `23-26` | score `0.8203` | text ` the clinical note while`
- auxiliary revision: matches `1`, mean alt prob `0.5989`, revise gain `+1`, retire delta `-1`

### entity_property_conflict
- span `25-25` | mean `1.0000` | max `1.0000` | text `-line`
- span `22-23` | mean `0.9336` | max `0.9727` | text ` the routine`
- span `29-30` | mean `0.9336` | max `0.9727` | text ` the clinical`
- span `27-27` | mean `0.9102` | max `0.9102` | text `.`
- active anchor spans: `26-27`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `1.0000`
- proposal-like future hint spans:
  - `25-25` | mean `1.0000` | text `-line`
  - `22-23` | mean `0.9336` | text ` the routine`
  - `29-30` | mean `0.9336` | text ` the clinical`
- auxiliary proposals:
  - `25-25` | score `1.0000` | text `-line`
  - `22-23` | score `0.9336` | text ` the routine`
  - `29-30` | score `0.9336` | text ` the clinical`
- auxiliary revision: matches `2`, mean alt prob `0.4695`, revise gain `+1`, retire delta `-1`

### legal_scope_stable
- span `18-18` | mean `0.9648` | max `0.9648` | text `.`
- span `14-16` | mean `0.8711` | max `0.9336` | text ` for non-commercial`
- span `20-24` | mean `0.8672` | max `1.0000` | text ` the clause while preserving those`
- span `12-12` | mean `0.8438` | max `0.8438` | text ` and`
- active anchor spans: `8-9, 17-18, 26-27`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `0.3333`
- proposal-like future hint spans:
  - `14-16` | mean `0.8711` | text ` for non-commercial`
  - `20-24` | mean `0.8672` | text ` the clause while preserving those`
- auxiliary proposals:
  - `14-16` | score `0.8711` | text ` for non-commercial`
  - `20-24` | score `0.8672` | text ` the clause while preserving those`
- auxiliary revision: matches `2`, mean alt prob `0.2759`, revise gain `+2`, retire delta `+0`

### legal_scope_conflict
- span `29-29` | mean `1.0000` | max `1.0000` | text ` commercial`
- active anchor spans: `8-9, 35-36`
- future-span overlap ratio: `0.0000` | anchor-span overlap ratio: `0.0000`
- proposal-like future hint spans:
  - `29-29` | mean `1.0000` | text ` commercial`
- auxiliary proposals:
  - `29-29` | score `1.0000` | text ` commercial`
- auxiliary revision: matches `1`, mean alt prob `0.2075`, revise gain `+1`, retire delta `-1`

### units_stable
- span `19-19` | mean `0.8984` | max `0.8984` | text `.`
- span `21-24` | mean `0.8398` | max `1.0000` | text ` the calculation notes while`
- span `26-28` | mean `0.8164` | max `0.8828` | text ` the same unit`
- active anchor spans: `2-3, 19-20`
- future-span overlap ratio: `0.3333` | anchor-span overlap ratio: `0.5000`
- proposal-like future hint spans:
  - `21-24` | mean `0.8398` | text ` the calculation notes while`
  - `26-28` | mean `0.8164` | text ` the same unit`
- auxiliary proposals:
  - `21-24` | score `0.8398` | text ` the calculation notes while`
  - `26-28` | score `0.8164` | text ` the same unit`
- auxiliary revision: matches `2`, mean alt prob `0.5019`, revise gain `+1`, retire delta `-1`

### units_conflict
- span `36-36` | mean `0.9648` | max `0.9648` | text ` meters`
- span `32-33` | mean `0.9609` | max `0.9922` | text ` were already`
- span `42-43` | mean `0.9414` | max `1.0000` | text ` the calculation`
- span `40-40` | mean `0.8984` | max `0.8984` | text `.`
- active anchor spans: `2-3, 13-14, 40-41`
- future-span overlap ratio: `0.2500` | anchor-span overlap ratio: `0.3333`
- proposal-like future hint spans:
  - `36-36` | mean `0.9648` | text ` meters`
  - `32-33` | mean `0.9609` | text ` were already`
  - `42-43` | mean `0.9414` | text ` the calculation`
- auxiliary proposals:
  - `36-36` | score `0.9648` | text ` meters`
  - `32-33` | score `0.9609` | text ` were already`
  - `42-43` | score `0.9414` | text ` the calculation`
- auxiliary revision: matches `2`, mean alt prob `0.4883`, revise gain `+2`, retire delta `-1`

## Interpretation

- This report is an experimental midpoint between delta-hidden heuristics and full leave-one-out KL.
- Scores are based on gradient influence of token positions on a future autoregressive loss window.
- High-scoring positions are candidates for semantically important context even when local hidden-state jumps are ambiguous.
- Grouped high-influence spans help test whether future-attribution concentrates on the same regions as current active anchors or highlights missed context spans.
