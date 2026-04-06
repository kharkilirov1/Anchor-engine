# AAHSA PoC Plan

**Date**: 2026-04-06
**Goal**: доказать что flatness-driven adaptive routing улучшает retention vs. fixed policy
**Model**: frozen Qwen (1.5B или 3.5-4B), без обучения

---

## Phase 1 — Adaptive compute routing (самый реализуемый кусок)

**Задача**: flat-якорь → расширить attention budget, mature/template → сузить.

### Что делаем

1. **Модифицируем attention scores на forward pass** (не архитектуру):
   - Hook в `model.layers[L].self_attn` для target layers
   - Перед softmax: добавить anchor-aware bias к attention scores
   - `attn_scores[i, j] += γ · μ_a · ω_aj` для токенов в anchor span
   - Для flat-якорей: не зажимать attention window, дать модели смотреть шире

2. **Три политики routing** (сравниваем):
   - `fixed_narrow`: attention bias только на anchor span (текущее поведение)
   - `adaptive_flat_wide`: flat → bias на расширенную окрестность anchor span
   - `adaptive_mature_skip`: mature → убираем bias полностью

3. **Метрика**:
   - Retention quality score (positive - negative keywords) на 500+ токенов
   - Сравнение: adaptive routing vs always-anchor vs always-base

### Файлы

- `src/model/aahsa_attention_hook.py` — hook для модификации attention scores
- `scripts/run_aahsa_poc_phase1.py` — прогон по 20 доменам
- Результаты в `archive/aahsa_poc/`

### Критерий успеха Phase 1

- adaptive routing даёт **меньше LOSS** чем always-anchor (текущий baseline: 6 LOSS с auto-calibration)
- flat-домены: anchor помогает (WIN или SAME)
- mature-домены: отсутствие anchor не вредит (SAME или лучше)

---

## Phase 2 — PCA direction вместо mean delta

**Задача**: заменить `d_a = mean(Δh)` на `d_a = SVD_1(Δh)` (первый сингулярный вектор).

### Что делаем

1. В `anchor_geometry.py` добавить `compute_principal_direction(delta_vectors)`:
   - `U, S, V = torch.linalg.svd(delta_vectors)`
   - `d_a = V[0]` (первая правая сингулярная вектор-строка)
   - Уже считаем `svdvals` — нужен полный SVD

2. Сравнить `d_a_mean` vs `d_a_pca` как сигнал:
   - Косинус между `d_a` и delta суффикса (carryover direction)
   - Корреляция с quality score

### Критерий успеха Phase 2

- PCA direction даёт более стабильный carryover signal чем mean delta
- Особенно для flat-якорей, где mean delta ≈ шум

---

## Phase 3 — Trajectory alignment в attention (главный научный эксперимент)

**Задача**: проверить `cos(V_Δ · Δh_t, d_a)` как attention signal.

### Что делаем

1. Для каждого токена t в контексте, считаем:
   - `Δh_t = h_t - h_{t-1}` (локальное движение)
   - `score_t = cos(Δh_t, d_a)` (совпадение с траекторией якоря)

2. Добавить `score_t` как дополнительный bias в attention:
   - `attn_scores[i, t] += β · score_t` для позиций t

3. Сравнить с Phase 1 baseline:
   - Phase 1 routing + trajectory alignment vs Phase 1 routing alone

### Критерий успеха Phase 3

- Trajectory alignment даёт дополнительное улучшение поверх adaptive routing
- Особенно на длинных генерациях (>300 токенов) где drift наиболее вероятен

---

## Порядок выполнения

```
Phase 1 (adaptive compute)     → доказать что flatness routing работает
    ↓
Phase 2 (PCA direction)        → улучшить d_a, можно параллельно с Phase 1
    ↓
Phase 3 (trajectory alignment) → только после стабильного Phase 1
```

---

## Ограничения

- **Frozen model only** — не трогаем веса, только hooks / score injection
- **Inference-time** — никакого training в PoC
- **20-domain benchmark** — тот же набор что в geometry-gated campaign
- **Не replacement attention** — AAHSA работает как controller поверх стандартного attention

---

## Связь с существующим кодом

| Компонент | Уже есть | Нужно |
|-----------|----------|-------|
| r1 / μ_a | `compute_geometry_metrics` | — |
| d_a (mean) | `compute_mean_direction` | PCA variant |
| Cluster classification | `auto_calibrate_thresholds` | — |
| Anchor span matching | `match_anchor_span` + decoded_pieces fallback | — |
| Attention hook | — | `aahsa_attention_hook.py` |
| Block overlap | — | простой, ~20 строк |
| ω_aj (block-anchor overlap) | — | простой, ~10 строк |
| Trajectory alignment score | — | `cos(Δh_t, d_a)` per token |
