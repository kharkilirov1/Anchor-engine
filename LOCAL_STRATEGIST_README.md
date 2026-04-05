# 🤖 Локальный AI-стратег для ABPT Orchestrator

Этот модуль позволяет заменить внешние API (DeepSeek, Anthropic, OpenAI) на локального AI-агента для принятия решений о следующих экспериментах.

## Быстрый старт

### 1. Установка

Файлы уже созданы:
- `scripts/local_strategist.py` — модуль локального стратега
- `scripts/orchestrate.py` — обновлённый оркестратор с поддержкой локального режима

### 2. Запуск из командной строки

```bash
# Вариант 1: Через переменную окружения
export USE_LOCAL_STRATEGIST=1
python scripts/orchestrate.py --budget 3 --llm-strategist

# Вариант 2: Через флаг
python scripts/orchestrate.py --budget 3 --local-strategist

# Вариант 3: Автоматический режим (без интерактива)
export USE_LOCAL_STRATEGIST=1
export LOCAL_STRATEGIST_MODE=auto
python scripts/orchestrate.py --budget 3 --llm-strategist
```

### 3. Запуск из Jupyter/Colab

Замените ячейку запуска orchestrator на:

```python
import os
import subprocess
import sys

os.environ["USE_LOCAL_STRATEGIST"] = "1"
os.environ["LOCAL_STRATEGIST_MODE"] = "interactive"

cmd = [
    sys.executable, 
    "scripts/orchestrate.py", 
    "--budget", "3",
    "--local-strategist"
]

result = subprocess.run(cmd, cwd="/content/ABPT")
```

## Режимы работы

### Interactive (по умолчанию)

Оркестратор выводит prompt для стратега и использует встроенную эвристику для выбора эксперимента. В этом режиме вы можете:
- Скопировать prompt и отправить своему AI-ассистенту вручную
- Использовать интеграцию с Kimi CLI (см. ниже)
- Подменить ответ, создав файл `/tmp/ABPT_strategist_response.json`

### Auto (автоматический)

```bash
export LOCAL_STRATEGIST_MODE=auto
```

В этом режиме локальный стратег пытается автоматически вызвать:
1. **Ollama** (`ollama run qwen2.5:14b`)
2. **llama.cpp server** (localhost:8080)
3. **vLLM** или другой OpenAI-compatible API

## Интеграция с Kimi CLI

### Вариант 1: Прямая интеграция (если Kimi доступен как команда)

Отредактируйте функцию `_call_local_agent()` в `scripts/local_strategist.py`:

```python
def _call_local_agent(context: dict[str, Any]) -> str | None:
    prompt = _build_strategist_prompt(context)
    
    # Вызов Kimi CLI
    import subprocess
    result = subprocess.run(
        ["kimi", "--raw", prompt],
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout
```

### Вариант 2: Ручной режим (рекомендуется)

1. Запустите orchestrator с локальным стратегом
2. Скопируйте prompt из вывода
3. Отправьте его Kimi
4. Сохраните ответ в `/tmp/ABPT_strategist_response.json`
5. Orchestrator продолжит работу

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestrator                            │
│                         │                                   │
│         ┌───────────────┴───────────────┐                   │
│         │                               │                   │
│         ▼                               ▼                   │
│  ┌──────────────┐              ┌────────────────┐          │
│  │ API Strategist│              │ Local Strategist│          │
│  │  (DeepSeek)   │              │   (Kimi/Local)  │          │
│  └──────────────┘              └────────────────┘          │
│                                         │                   │
│                              ┌─────────┴─────────┐         │
│                              ▼                   ▼         │
│                    ┌─────────────────┐  ┌──────────────┐   │
│                    │  Interactive    │  │    Auto      │   │
│                    │  (heuristic)    │  │  (Ollama/etc)│   │
│                    └─────────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Тестирование

```bash
# Тест локального стратега
python -c "
from scripts.local_strategist import strategist_local_select
import json

state = {
    'budget_remaining': 5,
    'current_phase': 1,
    'phases': {'1': {'experiments': []}},
    'known_facts': {},
    'open_hypotheses': []
}
playbook = ''

result = strategist_local_select(state, playbook)
print(json.dumps(result, indent=2))
"
```

## Отличия от API-стратега

| Функция | API Strategist | Local Strategist |
|---------|---------------|------------------|
| Требует интернет | Да | Нет |
| Стоимость | $$$ | Бесплатно |
| Задержка | 1-3 сек | Мгновенно (heuristic) или зависит от модели |
| Качество | Высокое (GPT-4/Claude) | Зависит от локальной модели |
| Новые скрипты | Может генерировать | Требует manual mode или auto с LLM |

## Troubleshooting

### "local_strategist.py не найден"

```bash
# Проверьте что файл существует
ls -la scripts/local_strategist.py

# Проверьте Python path
python -c "import sys; print(sys.path)"
```

### "Ответ не JSON"

Локальный агент вернул неправильный формат. Убедитесь, что:
- Ответ содержит только JSON (без markdown ```json)
- Все обязательные поля присутствуют: `id`, `description`, `script`

### "Все эксперименты выполнены"

Локальный стратег использует эвристику и не придумывает новые гипотезы. Для генерации новых идей используйте:
- API-стратега (DeepSeek)
- Ручной режим с собственным AI
- Редактирование `EXPERIMENT_REGISTRY` в `orchestrate.py`

## Roadmap

- [ ] Поддержка WebSocket для real-time интерактива
- [ ] Интеграция с Ollama API
- [ ] Кэширование ответов стратега
- [ ] Визуальный интерфейс выбора экспериментов
