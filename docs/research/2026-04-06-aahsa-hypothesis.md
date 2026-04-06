# AAHSA — Anchor-Aware Hierarchical Sparse Attention

**Date**: 2026-04-06
**Author**: @kharkilirov1
**Status**: hypothesis (pre-PoC)

---

## Core idea

Attention routing определяется не только content similarity, а тремя дополнительными сигналами:

1. **Anchor semantics** — близость запроса к смысловому центру якоря
2. **Trajectory geometry** — совпадение локального движения токена с направлением сборки якоря
3. **Adaptive compute** — ширина поиска зависит от зрелости/плоскости якоря

---

## Mathematical formulation

### Anchor definitions

Для каждого якоря `a` с span `S_a`:

- **Center**: `c_a = (1/|S_a|) Σ_{t ∈ S_a} h_t`
- **Principal direction**: `d_a` = первый сингулярный вектор SVD на `{h_t - h_{t-1} : t ∈ S_a}`
  (не mean delta — PCA direction робастнее, особенно для flat-якорей)
- **Maturity**: `μ_a ∈ [0,1]` — rank1 explained variance (r1)
- **Flatness**: `f_a = 1 - μ_a`
- **Contradiction pressure**: `p_a ∈ [0,1]`

Mapping на кластеры:
- template → μ_a высокое, f_a низкое
- mature → μ_a средне-высокое
- flat → μ_a низкое, f_a высокое

### Block-level scoring

Контекст разбит на блоки `j = 1..N`, размер блока `B`, overlap 1 токен.
Pooled representation блока: `b_j = Pool({h_t : t ∈ block_j})`.

Block score:

```
R_ij = α · (q_i^T W_b b_j / √d)
     + β · Σ_a ω_aj · cos(U_q q_i, U_c c_a)
     + γ · Σ_a ω_aj · μ_a
     - δ · Σ_a ω_aj · p_a
```

Где `ω_aj = |{t ∈ block_j : t ∈ anchor_a}| / |block_j|` — доля токенов блока в якоре.

Top-m_i блоков: `B_i = TopM({R_ij})`.

### Token-level scoring

Внутри выбранных блоков:

```
r_it = α' · (q_i^T k_t / √d)
     + β' · Σ_a η_at · cos(V_Δ (h_t - h_{t-1}), d_a)
     + γ' · Σ_a η_at · μ_a
     - δ' · Σ_a η_at · p_a
```

Где `η_at` — близость токена t к якорю a.

Второй член — **trajectory geometry alignment**: не "на что это похоже", а "двигает ли это представление в правильном направлении".

Top-k_i токенов → sparse attention только по ним.

### Adaptive compute

Эффективная неопределённость для токена i:

```
u_i = Σ_a ρ_ia · f_a
```

Бюджет поиска:

```
m_i = m_min + ⌈λ_m · u_i⌉    (блоки)
k_i = k_min + ⌈λ_k · u_i⌉    (токены)
```

- template → малый бюджет (модель на рельсах)
- mature → средний бюджет
- flat → широкий поиск, возможно branching

---

## Novelty

Стандартный hierarchical sparse attention (HISA, BigBird, Longformer):

```
importance ≈ similarity
```

AAHSA:

```
importance ≈ similarity + anchor semantics + trajectory geometry + state maturity
```

Trajectory alignment (`cos(V_Δ · Δh, d_a)`) — не сводится к существующим подходам (MoE, early exit, sparse attention). Это новый сигнал для attention routing.

---

## Complexity

Per-query: `O(L/B + m_i · B)` вместо `O(L)`.
Anchor overhead: `O(|A|)` на query — пренебрежимо.
Субквадратично при `m_i ≪ L/B`.

---

## Consolidated verdict (по 3 независимым LLM-обзорам + мета-анализ)

**Подтверждено** (можно использовать):
- Кластеризация якорей (mature/template/flat) — измерена на 20 доменах
- Adaptive compute по flatness — логична, тестируема, самый реализуемый кусок
- Trajectory geometry как сигнал — новый, не тривиальный, самый сильный теоретически

**Гипотеза** (нужен PoC):
- AAHSA как полный replacement attention → слишком сложно для MVP
- AAHSA как routing controller для существующего attention → реалистично

**Известные инженерные проблемы**:
1. Граница блока — первый токен не имеет h_{t-1} из предыдущего блока → overlap на 1 токен
2. TopM/TopK недифференцируемы → ok для inference, Gumbel-softmax для training
3. μ_a — свойство спана, broadcast на токены, нет intra-span градации
4. α,β,γ,δ — потенциальная точка нестабильности без калибровки
5. Inter-anchor conflict не формализован

**Consensus всех обзоров**: inference-only PoC на frozen model → первый шаг.

---

## References

- Review: `docs/research/2026-04-06-aahsa-review.md` (Claude Opus 4.6)
- External reviews: Gemini, Kimi (не в репо, устный анализ)
- Experimental basis: 20-domain retention campaign, geometry-gated results
- Codebase: `src/utils/anchor_geometry.py`, `src/model/qwen_anchor_overlay.py`
