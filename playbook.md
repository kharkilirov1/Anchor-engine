# ABPT Research Playbook
> Накопленные знания. Обновляется автоматически после каждого эксперимента.
> Формат: факт → источник → статус (confirmed / hypothesis / disproven)

---

## Модель

- **Qwen/Qwen3.5-4B** — 32 слоя, hidden_dim=2048
- **Qwen/Qwen2.5-1.5B** — 28 слоёв, нет чёткой crystallization zone

---

## Геометрия hidden states

### Фазовые зоны (Qwen3.5-4B)
| Зона | Слои | Описание | Статус |
|------|------|----------|--------|
| Pre-semantic | L0–L3 | r1 не растёт, случайная геометрия | confirmed |
| **Crystallization** | **L4–L8** | резкий рост r1, anchor concept формируется | confirmed |
| Propagation | L9–L15 | стабилизация, интеграция с контекстом | confirmed |
| Integration | L16–L23 | снижение r1, подготовка к генерации | confirmed |
| Handoff | L24–L31 | передача в generation head | confirmed |

### Group-specific carryover layers
Carryover = косинусное сходство между hidden state суффикса и concept vector anchor span.
Пик carryover — слой где концепт максимально влияет на продолжение.

| Группа | Carryover слой | Стабильность | Пиковое значение |
|--------|---------------|--------------|-----------------|
| json_only_response_format_policy | L11 | ✓ стабильно | 0.156 |
| proof_by_contradiction_reasoning_steps | L25 | ✓ стабильно | 0.125 |
| dependency_injection_request_flow_sequence | L24 | ✓ стабильно | 0.168 |
| binary_search_update_loop_procedure | L10 | ✓ стабильно | 0.125 |
| strictly_vegan_meal_plan_policy | L11 | ✓ стабильно | 0.107 |
| async_fastapi_service_architecture_policy | UNSTABLE | ✗ нестабильно | — |

### Anchor mechanism
- **Гипотеза:** anchor = attention beacon, не rewrite hidden states
- Carryover probe: near-zero delta в suffix hidden states → подтверждает beacon
- Статус: **partially_supported** (geometry probe after tokenization controls дал `partially_supported`; нужна paraphrase/attention верификация)

---

## Anchor профили

| Профиль | Длина anchor | Concept norm | Вывод |
|---------|-------------|--------------|-------|
| short | 3–5 токенов | ~68 | слабый сигнал |
| **medium** | 5–6 токенов | ~64 | **оптимальный** |
| long | 8–9 токенов | ~58 | деградация сигнала |

**Правило:** длиннее ≠ лучше. Оптимум ≈ 5–6 токенов.

---

## Кластеры (calibration)

Три кластера на основе r1@mature_layer:
- **flat** — r1 < 0.469 → base generation слабая, rescue возможен
- **mature** — r1 ≥ 0.469 → base generation устойчивая
- **template** — любой r1 + template_delta < −0.004 → специфичная геометрия

**Предупреждение:** граница flat/mature razor-thin (0.469 vs 0.470). Порог нестабилен на малом наборе кейсов.

---

## Политики routing

| Политика | Wins | Rescue rate | Activation rate | Рекомендация |
|----------|------|-------------|-----------------|--------------|
| always_base | — | — | 0% | baseline |
| always_anchor | — | — | 100% | плохо |
| **flat_failure_gated** | 1 | 1.0 | 7.7% | **оптимально** |
| failure_gated_any | 1 | 1.0 | 23% | хуже (лишние активации) |

---

## Известные провалы

- **procedure_contradiction_proof** — anchor активно вредит (base=1, anchor=0, delta=−1)
  - Кластер: template. Причина: не диагностирована.
- **Anchor sweep (все три профиля)** — wins=0, rescue_rate=0.0
  - Причина: anchor spans 5–8 токенов → signal dilution; reference layer сдвинулся на L28

---

## Что НЕ работает
- Поиск universal threshold по одному reference layer → нестабильно
- Длинные anchor spans (8–9 токенов) → норма падает, сигнал деградирует
- Запуск calibration без зафиксированных probe_layers → результат меняется

---

## Последний прогон (2026-04-04)
- `early_slope_4_8` снова не подтвердился: `ρ = -0.1449`
- `tail_retention_ratio` остаётся сильнейшим сигналом: `ρ = +0.6419`
- `group_routing_vs_universal_threshold` дал только слабый положительный эффект: `mean_last_token_delta = 0.0056`
- `attention_beacon_crystallization_zone` вернул `partially_supported`, то есть гипотеза жива, но не закрыта

---

## Открытые вопросы
1. Сохраняется ли сила `tail_retention_ratio` на short/long profiles, а не только на medium? (Фаза 1)
2. Может ли group-specific routing показать **материальный** эффект выше текущего слабого `mean_last_token_delta=0.0056`? (Фаза 2)
3. Отличается ли геометрия injection-anchor от легитимного anchor? (Фаза 3)
4. Почему fastapi группа нестабильна в carryover?
5. Почему contradiction_proof кейс даёт отрицательный delta?

---

## Bugs исправленные
- `build_calibration_summary()` не получала `thresholds=` аргумент → null в JSON (commit 45b9682, исправлено)

---

*Последнее обновление: 2026-04-04 (ручное)*
*Следующее обновление: автоматически после Фазы 1 phase_probe*

---
## [2026-04-04] Эксперимент H1_cross_profile_v2: Retry cross-profile tail_retention_ratio validation across short/medium/long profiles in a single run. All 8 prior experiments returned null metrics due to infrastructure/extraction failures, not hypothesis rejection. This is the highest-value experiment: confirms or denies generalization of the strongest known signal (rho=0.64 on medium) with one run covering all 3 profiles.

**Статус:** 📊 DATA COLLECTED  
**Метрика:** `cross_profile.medium.tail_retention_rho` = `None`  

---
## [2026-04-04] Эксперимент H1_per_case_diagnostic: Minimal per-case tail_retention_ratio + constraint_score diagnostic on medium profile. Validates rho=0.64 independently and produces raw per-case data. Ultra-robust: flat output, no nested metric paths, aggressive try/except per case.

**Статус:** 📊 DATA COLLECTED  
**Метрика:** `spearman_rho` = `None`  

---
## [2026-04-04] Эксперимент infra_validate_diagnostic: Infrastructure validation via simplest script (per_case_diagnostic). All 9 prior experiments returned null metrics despite successful execution (14-62s runtime). Before spending more budget on new hypotheses, we must confirm the result extraction pipeline works. This script outputs ===FINAL_RESULT=== with flat JSON — if this also returns null, the bug is in the orchestrator's metric parser, not in experiment logic.

**Статус:** 📊 DATA COLLECTED  
**Метрика:** `spearman_rho` = `None`  

---
## [2026-04-04] Эксперимент H1_short_diagnostic: Per-case tail_retention diagnostic on SHORT anchor profile. Tests whether rho=0.64 (medium) generalizes to shorter anchors. Uses proven per_case_diagnostic approach with added --profile arg. Flat output, no nested paths, maximal robustness.

**Статус:** 📊 DATA COLLECTED  
**Метрика:** `spearman_rho` = `None`  

---
## [2026-04-05] Эксперимент H1_short_diagnostic_v2_retry: Проверить, сохраняется ли signal tail_retention_ratio на short anchor profile самым робастным flat-output скриптом, избегая тяжёлых multi-profile прогонов и nested result extraction.

**Статус:** 📊 DATA COLLECTED  
**Метрика:** `spearman_rho` = `None`  

---
## [2026-04-05] Эксперимент H1_medium_diagnostic_v2_cpu_repro: Лёгкий CPU-only repro на medium profile через flat-output diagnostic, чтобы проверить, воспроизводится ли ранее наблюдавшийся medium tail-retention signal и отличить реальный null от проблем извлечения метрики.

**Статус:** ✅ CONFIRMED  
**Метрика:** `spearman_rho` = `0.6`  

---
## [2026-04-05] Эксперимент H1_long_diagnostic_v2_cpu_compare: Проверить, сохраняется ли per-case tail-retention signal на long anchor profile, чтобы напрямую сравнить short/medium/long через самый робастный CPU-safe flat diagnostic и уточнить, действительно ли medium является оптимальным профилем.

**Статус:** ❌ NOT CONFIRMED  
**Метрика:** `spearman_rho` = `0.257143`  

---
## [2026-04-05] Эксперимент H1_short_diagnostic_v2_cpu_repro: CPU-only flat diagnostic on short anchor profile to resolve the remaining short-vs-medium uncertainty with the most robust existing script and avoid known broken geometry/injection paths.

**Статус:** ❌ NOT CONFIRMED  
**Метрика:** `spearman_rho` = `-0.3`  

---
## [2026-04-05] Эксперимент H3_injection_geometry_cpu_v1: Test whether injected (out-of-context) anchor spans produce geometrically distinguishable hidden state trajectories vs legitimate anchors. Core Phase 3 question. Uses medium profile (confirmed strongest) with minimal case cap for CPU safety.

**Статус:** ✅ CONFIRMED  
**Метрика:** `summary.detection_auc` = `1.0`  

---
## [2026-04-05] Эксперимент H7_carryover_contradiction_medium_cpu_v2: Diagnose WHY contradiction_proof case shows negative delta (anchor hurts, base=1 anchor=0 delta=-1). Run carryover probe on just this case with medium profile to see if carryover signal is absent, inverted, or misrouted to wrong layer. Previous H4 attempt returned null metric — likely infra extraction issue, not hypothesis failure.

**Статус:** ❌ NOT CONFIRMED  
**Метрика:** `summary.mean_last_token_delta` = `0.014798829881328857`  

---
## [2026-04-05] Эксперимент H3_injection_geometry_robustness_cap4: Validate injection geometry detection AUC=1.0 with 4 cases per group instead of 2. The AUC=1.0 result from H3_injection_geometry_cpu_v1 is the strongest Phase 3 finding but was tested on minimal data (cap=2). If AUC stays high with doubled sample, the injected-vs-legitimate anchor discriminator is confirmed robust. If it drops, we know the detection limit.

**Статус:** ✅ CONFIRMED  
**Метрика:** `summary.detection_auc` = `1.0`  
