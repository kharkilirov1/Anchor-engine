# Qwen Signal Proxy Comparison

Date: 2026-03-29 05:34 UTC
Anchor probe source: `Qwen/Qwen2.5-1.5B`
Future-influence source: `Qwen/Qwen2.5-1.5B`

## Summary

- Families compared: `8`
- Delta-pressure wins: `5`
- Delta-viability wins: `5`
- Mean future-influence wins: `3`
- Anchor-position future-influence wins: `5`
- Mean pressure gap: `0.0389`
- Mean viability gap: `-0.0260`
- Mean future-influence gap: `-0.0250`
- Mean anchor-position future-influence gap: `0.2610`

## Family table

| Family | Pressure gap | Viability gap | Mean future gap | Anchor-position future gap | Delta joint | Future anchor |
|---|---:|---:|---:|---:|---|---|
| api_framework | -0.0044 | 0.0038 | 0.0586 | 0.8398 | miss | win |
| entity_property | 0.1004 | -0.0476 | 0.0312 | 0.9102 | win | win |
| induction | 0.0759 | -0.1896 | -0.0117 | 0.5378 | win | win |
| instruction_constraints | 0.1290 | -0.1314 | -0.0879 | -0.3210 | win | miss |
| legal_scope | 0.0330 | -0.0882 | -0.2021 | -0.3216 | win | miss |
| proof_mode | 0.0884 | -0.0383 | 0.1348 | 0.4336 | win | win |
| quantifier | -0.0755 | 0.1890 | -0.0312 | 0.1251 | miss | win |
| units | -0.0357 | 0.0943 | -0.0918 | -0.1159 | miss | miss |

## Interpretation

- The existing delta-based anchor diagnostics still provide the cleanest global stable-vs-conflict signal.
- Raw mean future influence is mixed and should not yet be treated as a drop-in replacement for contradiction/viability scoring.
- Anchor-position future influence looks more promising than plain prompt-level future influence, because it wins in more families and highlights conflict-sensitive positions even when prompt-level means are ambiguous.
- The current best reading is that future-conditioned attribution is useful as a positional relevance probe, not yet as a single scalar anchor score.
