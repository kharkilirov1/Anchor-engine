"""
Ячейка для ABPT_Research_Campaign.ipynb — запуск с локальным стратегом.

Замените ячейку запуска orchestrator на содержимое этого файла.
"""

# ═════════════════════════════════════════════════════════════════════════════
# Ячейка 15 (замена): Запуск Orchestrator с локальным AI-стратегом
# ═════════════════════════════════════════════════════════════════════════════

import os
import subprocess
import sys

BUDGET_THIS_SESSION = 3

# Принудительно включаем локального стратега
os.environ["USE_LOCAL_STRATEGIST"] = "1"
os.environ["STRATEGIST_BACKEND"] = "codex"
os.environ["STRATEGIST_CODEX_MODE"] = "autonomous"
os.environ["LOCAL_STRATEGIST_MODE"] = "interactive"  # или "auto" для автоматического режима

print("=" * 60)
print("🤖 ABPT Orchestrator с локальным Codex-стратегом")
print("=" * 60)
print(f"Бюджет экспериментов: {BUDGET_THIS_SESSION}")
print("Режим: LOCAL (локальный агент)")
print("=" * 60)

# Проверяем наличие локального стратега
try:
    from scripts.local_strategist import strategist_local_select
    print("✅ Модуль local_strategist.py найден")
except ImportError as e:
    print(f"❌ Ошибка импорта local_strategist.py: {e}")
    print("Убедитесь, что файл scripts/local_strategist.py существует")
    raise

# Запускаем оркестратор с флагом --local-strategist
cmd = [
    sys.executable, 
    "scripts/orchestrate.py", 
    "--budget", str(BUDGET_THIS_SESSION),
    "--local-strategist"
]

result = subprocess.run(cmd, cwd="/content/ABPT")

# ═════════════════════════════════════════════════════════════════════════════
# Альтернативный вариант: Ручной режим (для отладки)
# ═════════════════════════════════════════════════════════════════════════════
"""
# Если нужно вручную указать следующий эксперимент без AI:

cmd = [
    sys.executable,
    "scripts/orchestrate.py",
    "--budget", str(BUDGET_THIS_SESSION),
    "--experiment", "H1_short"  # указать конкретную гипотезу
]
result = subprocess.run(cmd, cwd="/content/ABPT")
"""

# ═════════════════════════════════════════════════════════════════════════════
# Дополнительная ячейка: Просмотр текущего состояния
# ═════════════════════════════════════════════════════════════════════════════

import json
from pathlib import Path

state_path = Path("/content/ABPT/research_state.json")
if state_path.exists():
    state = json.loads(state_path.read_text())
    print("\n" + "=" * 60)
    print("📊 Текущее состояние кампании")
    print("=" * 60)
    print(f"Фаза: {state.get('current_phase', 'N/A')}")
    print(f"Остаток бюджета: {state.get('budget_remaining', 0)}/{state.get('budget_max', 0)}")
    print(f"Выполнено экспериментов: {state.get('experiments_run', 0)}")
    
    # Показываем последние метрики
    history = state.get('metric_history', [])
    if history:
        print("\nПоследние результаты:")
        for h in history[-3:]:
            print(f"  - {h.get('experiment')}: {h.get('metric')} = {h.get('value')}")
