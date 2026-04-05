# Agent Task: run_qwen_geometry_generation_calibration.py

## Цель

Написать `scripts/run_qwen_geometry_generation_calibration.py` в проекте
`/mnt/c/Users/Kharki/Desktop/ABPT`.

Это калибровочный эксперимент: для каждого anchor-кейса — один форвард промпта
с извлечением rank1-профиля по слоям, затем генерация 120 токенов, затем
constraint-анализ. Результат: таблица
`(кейс, anchor_cluster, rank1_profile, constraint_score)` для калибровки
routing-порогов.

---

## Архитектура проекта

Строго следуй паттернам из двух существующих скриптов:

- `scripts/run_qwen_anchor_geometry_probe.py` — форвард с
  `output_hidden_states=True`, извлечение hidden states по слоям,
  `compute_geometry_metrics`, `match_anchor_span`.
- `scripts/run_qwen_long_retention_compare.py` — `generate_base()`,
  `analyze_keywords()`, `overlay.generate_with_anchor_bias()`.

Используй те же импорты:

```python
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import (
    compute_geometry_metrics,
    extract_delta_vectors,
    match_anchor_span,
    decode_token_pieces,
    decode_token_surfaces,
    token_has_leading_whitespace,
)
from src.data.qwen_anchor_geometry_cases import (
    make_qwen_anchor_geometry_cases,
    QwenAnchorGeometryCase,
)
```

Не переписывай существующие утилиты — переиспользуй.

---

## Что делает скрипт

Для каждого кейса из `make_qwen_anchor_geometry_cases()`:

### Шаг 1 — Geometry Profile (один форвард промпта)

- Токенизировать промпт (`return_offsets_mapping=True`, `max_length=128`).
- Форвард с `output_hidden_states=True`, `torch.no_grad()`.
- `match_anchor_span` для нахождения позиции anchor в токенах.
- Для каждого слоя из `PROBE_LAYERS = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`:
  - `extract_delta_vectors(hidden_states[layer+1][0], token_start, token_end)`
  - `compute_geometry_metrics(delta_vectors)`
  - Сохранить `rank1_explained_variance` и `path_tortuosity`.
- Вычислить из профиля:
  - `r1_at_24` = rank1[layer=24]
  - `delta_l26_l27` = rank1[layer=27] − rank1[layer=26]
  - `slope_l18_l24` = линейный наклон rank1 по L18–L24 (7 точек,
    `numpy.polyfit` степень 1, возвращает коэффициент наклона)
- Классифицировать `anchor_cluster`:
  - `"mature"`   если `r1_at_24 > 0.65`
  - `"template"` если `delta_l26_l27 > 0.08`
  - `"flat"`     иначе

### Шаг 2 — Generation

- `generate_base()` — паттерн из `run_qwen_long_retention_compare.py`,
  `max_new_tokens=120`, `max_length=256`.
- `overlay.generate_with_anchor_bias()` с теми же параметрами и дефолтами,
  что в retention-скрипте.
- `analyze_keywords()` на `continuation_text` (см. маппинг ниже).

### Шаг 3 — Constraint Score

`constraint_score = quality_score` из `analyze_keywords` (уже считается
в retention-скрипте).  
Дополнительно: `drift_detected = anchor_analysis["negative_total"] > 0`.

---

## Keyword Mappings по кейсам

```python
KEYWORD_MAP = {
    "strictly_vegan_meal_plan_policy": {
        "positive": ["vegan","plant-based","tofu","lentil","chickpea",
                     "beans","vegetable","mushroom"],
        "negative": ["egg","eggs","cheese","butter","milk","cream",
                     "meat","chicken","beef"],
    },
    "async_fastapi_service_architecture_policy": {
        "positive": ["async","await","FastAPI","router","endpoint",
                     "dependency","asyncio"],
        "negative": ["Flask","Django","sync","blocking","thread","subprocess"],
    },
    "json_only_response_format_policy": {
        "positive": ["json","JSON","{","}","key","value","format"],
        "negative": ["markdown","plain text","prose","sorry",
                     "I cannot","Here is"],
    },
    "proof_by_contradiction_reasoning_steps": {
        "positive": ["assume","contradiction","suppose","therefore",
                     "absurd","QED","proof"],
        "negative": ["example","for instance","because","simply",
                     "just","obviously"],
    },
    "binary_search_update_loop_procedure": {
        "positive": ["mid","low","high","left","right","while",
                     "binary","O(log"],
        "negative": ["for i","linear","scan","iterate","brute"],
    },
    "dependency_injection_request_flow_sequence": {
        "positive": ["inject","dependency","container","resolve",
                     "provider","interface"],
        "negative": ["global","singleton","import","hardcode",
                     "direct instantiation"],
    },
}
```

Если группа кейса не в словаре — использовать пустые списки и пометить
`constraint_score = None`.

---

## Структура вывода JSON

Файл `archive/qwen_geometry_generation_calibration.json`:

```json
{
  "metadata": {
    "created_at_utc": "...",
    "model_name": "...",
    "device": "...",
    "max_length": 128,
    "max_new_tokens": 120,
    "probe_layers": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
  },
  "cases": [
    {
      "name": "...",
      "anchor_class": "...",
      "anchor_group": "...",
      "anchor_text": "...",

      "rank1_profile":      {"18": 0.0, "19": 0.0, "...": 0.0},
      "tortuosity_profile": {"18": 0.0, "19": 0.0, "...": 0.0},
      "r1_at_24": 0.0,
      "delta_l26_l27": 0.0,
      "slope_l18_l24": 0.0,
      "anchor_cluster": "mature|template|flat",

      "base_continuation": "...",
      "anchor_continuation": "...",

      "base_analysis": {
        "positive_total": 0,
        "negative_total": 0,
        "quality_score": 0.0,
        "drift_detected": false
      },
      "anchor_analysis": {
        "positive_total": 0,
        "negative_total": 0,
        "quality_score": 0.0,
        "drift_detected": false
      },

      "constraint_delta": 0.0
    }
  ],

  "calibration": {
    "by_cluster": {
      "mature":   {"n": 0, "mean_constraint_delta": 0.0,
                   "mean_drift_rate": 0.0, "r1_at_24_range": [0.0, 0.0]},
      "template": {"n": 0, "mean_constraint_delta": 0.0,
                   "mean_drift_rate": 0.0, "r1_at_24_range": [0.0, 0.0]},
      "flat":     {"n": 0, "mean_constraint_delta": 0.0,
                   "mean_drift_rate": 0.0, "r1_at_24_range": [0.0, 0.0]}
    },
    "threshold_candidates": {
      "r1_at_24_mature_threshold": 0.65,
      "delta_l26_l27_template_threshold": 0.08,
      "observed_separation": true
    }
  }
}
```

`observed_separation = True` если `flat.mean_constraint_delta < min(template, mature)`.

---

## Markdown отчёт

Файл `docs/research/qwen_geometry_generation_calibration.md`:

1. **Summary**: модель, дата, n_cases, количество кейсов в каждом кластере.
2. **Per-case таблица**:
   `name | cluster | r1@L24 | delta_L26→L27 | slope_L18-L24 | base_quality | anchor_quality | constraint_delta | drift_detected`
3. **Calibration summary**: `mean_constraint_delta` по кластерам,
   `observed_separation`.
4. **Вывод**: поддерживают ли данные пороги 0.65 / 0.08.

---

## Соображения реализации

- Модель грузится **один раз** (CPU, `Qwen/Qwen2.5-1.5B`).
- Geometry: `torch.no_grad()`, паттерн из `run_qwen_anchor_geometry_probe.py`.
- Generation: паттерн из `run_qwen_long_retention_compare.py`.
- `numpy` доступен (`polyfit` для slope).
- `seed = 7` для воспроизводимости (`torch.manual_seed`).
- `argparse`: `--model`, `--device` (default `cpu`), `--max_length` (128),
  `--max_new_tokens` (120), `--seed` (7), `--output_json`, `--output_md`.
- Прогресс: `print()` после каждого кейса с именем и кластером.
- Не перезаписывать существующие файлы — имя файла новое:
  `qwen_geometry_generation_calibration.json`.

---

## Ограничения

- **Не модифицировать** существующие файлы — только создать новый скрипт
  `scripts/run_qwen_geometry_generation_calibration.py`.
- Не создавать новые `src/` модули — только один скрипт в `scripts/`.
- Если `anchor_span` не найден — логировать `print(f"SKIP {case.name}: span not matched")`
  и пропускать без падения.
- Если `generate_with_anchor_bias` возвращает dict без `"continuation_text"` —
  адаптировать по аналогии с `run_qwen_long_retention_compare.py`.
- Не удалять и не трогать ничего в `archive/` кроме записи нового файла.
