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
