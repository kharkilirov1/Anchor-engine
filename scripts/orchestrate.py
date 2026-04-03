"""
ABPT Orchestrator — Autonomous Research Loop
=============================================
Читает research_state.json + playbook.md, запускает следующий эксперимент,
анализирует результат, обновляет state и playbook, повторяет.

Архитектура (Karpathy-style с AlphaLab playbook):
  Strategist → выбирает следующий эксперимент из open_hypotheses
  Worker     → запускает скрипт, собирает результат
  Analyzer   → читает JSON, обновляет state + playbook
  Loop       → повторяет до budget=0 или все гипотезы проверены

Использование (в Colab):
  !python scripts/orchestrate.py
  !python scripts/orchestrate.py --budget 5
  !python scripts/orchestrate.py --phase 1        # только фаза 1
  !python scripts/orchestrate.py --dry-run        # показать план без запуска
  !python scripts/orchestrate.py --experiment H1  # запустить конкретную гипотезу

Переменные окружения:
  OPENAI_API_KEY   — для LLM-assisted Strategist (опционально)
  ANTHROPIC_API_KEY — для LLM-assisted Strategist (опционально)
  Без API ключа: rule-based Strategist (работает автономно)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_FILE   = ROOT / "research_state.json"
PLAYBOOK_FILE = ROOT / "playbook.md"
ARCHIVE_DIR  = ROOT / "archive"
SCRIPTS_DIR  = ROOT / "scripts"
LOG_FILE     = ROOT / "orchestrator_log.jsonl"

SEPARATOR = "─" * 60


# ─────────────────────────────────────────────────────────────────────────────
# State I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict[str, Any]:
    return json.loads(STATE_FILE.read_text())


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def log_event(event: dict[str, Any]) -> None:
    event["timestamp"] = datetime.now(UTC).isoformat()
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Strategist — выбирает следующий эксперимент
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_REGISTRY: dict[str, dict[str, Any]] = {
    "H1": {
        "description": "Verify early_slope_4_8 predicts base_constraint_score",
        "phase": 1,
        "script": "run_qwen_phase_probe.py",
        "default_args": {"anchor_profile": "medium", "tau": "0.5"},
        "output_pattern": "archive/*_phase_probe_medium.json",
        "result_key": "correlation_summary.spearman_early_slope_4_8_vs_base_constraint",
        "success_threshold": 0.4,
        "depends_on": [],
    },
    "H1_short": {
        "description": "Phase probe with short anchor profile",
        "phase": 1,
        "script": "run_qwen_phase_probe.py",
        "default_args": {"anchor_profile": "short", "tau": "0.5"},
        "output_pattern": "archive/*_phase_probe_short.json",
        "result_key": "correlation_summary.spearman_early_slope_4_8_vs_base_constraint",
        "success_threshold": 0.4,
        "depends_on": ["H1"],
    },
    "H1_long": {
        "description": "Phase probe with long anchor profile",
        "phase": 1,
        "script": "run_qwen_phase_probe.py",
        "default_args": {"anchor_profile": "long", "tau": "0.5"},
        "output_pattern": "archive/*_phase_probe_long.json",
        "result_key": "correlation_summary.spearman_early_slope_4_8_vs_base_constraint",
        "success_threshold": 0.4,
        "depends_on": ["H1"],
    },
    "H4": {
        "description": "Attention corroboration — beacon hypothesis",
        "phase": 1,
        "script": "run_qwen_attention_corroboration_probe.py",
        "default_args": {"anchor_profile": "medium"},
        "output_pattern": "archive/*_attention_corroboration_*.json",
        "result_key": "summary.peak_attention_zone",
        "success_threshold": None,  # qualitative
        "depends_on": ["H1"],
    },
    "H2": {
        "description": "Group-specific routing vs universal threshold",
        "phase": 2,
        "script": "run_qwen_group_routing_probe.py",
        "default_args": {"anchor_profile": "medium"},
        "output_pattern": "archive/*_group_routing_*.json",
        "result_key": "summary.routing_delta_vs_flat_failure_gated",
        "success_threshold": 0.0,
        "depends_on": ["H1"],
    },
    "H3": {
        "description": "Injection detection via geometry anomaly",
        "phase": 3,
        "script": "run_qwen_injection_geometry_probe.py",
        "default_args": {"anchor_profile": "medium"},
        "output_pattern": "archive/*_injection_geometry_*.json",
        "result_key": "summary.detection_auc",
        "success_threshold": 0.7,
        "depends_on": ["H1"],
    },
}


def strategist_select_next(state: dict[str, Any], target_phase: int | None = None) -> str | None:
    """
    Rule-based Strategist.
    Выбирает следующую гипотезу по правилам:
    1. Не зависит от непроверенных гипотез
    2. Текущая фаза (или target_phase)
    3. Не была запущена ранее
    4. Бюджет не исчерпан
    """
    if state["budget_remaining"] <= 0:
        return None

    completed = {
        exp["hypothesis_id"]
        for phase_data in state["phases"].values()
        for exp in phase_data.get("experiments", [])
        if exp.get("status") in ("success", "failed", "done", "skipped")
    }

    phase_filter = target_phase or state["current_phase"]

    for hyp_id, hyp_def in EXPERIMENT_REGISTRY.items():
        if hyp_def["phase"] != phase_filter:
            continue
        if hyp_id in completed:
            continue
        deps = hyp_def.get("depends_on", [])
        if not all(dep in completed for dep in deps):
            continue
        return hyp_id

    # Если текущая фаза исчерпана — переходим к следующей
    if target_phase is None:
        next_phase = state["current_phase"] + 1
        if next_phase <= 6:
            print(f"[Strategist] Фаза {state['current_phase']} исчерпана → переход к Фазе {next_phase}")
            state["current_phase"] = next_phase
            return strategist_select_next(state, next_phase)

    return None


def strategist_llm_select(state: dict[str, Any], playbook: str) -> dict[str, Any] | None:
    """
    Free-form LLM Strategist.
    Предлагает следующий эксперимент в виде JSON — может придумать новую гипотезу.
    Возвращает dict с полями: id, description, script, args, result_key, success_threshold
    """
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not deepseek_key and not anthropic_key and not openai_key:
        return None

    # Список доступных скриптов
    scripts_dir = Path(__file__).resolve().parent
    available_scripts = [p.name for p in scripts_dir.glob("run_qwen_*.py")]

    completed = [
        {"id": exp["hypothesis_id"], "status": exp["status"],
         "metric": exp.get("metric_value")}
        for phase_data in state["phases"].values()
        for exp in phase_data.get("experiments", [])
    ]

    # Подсчёт использования скриптов — чтобы не зацикливаться
    script_usage: dict[str, int] = {}
    for phase_data in state["phases"].values():
        for exp in phase_data.get("experiments", []):
            s = exp.get("script", "")
            script_usage[s] = script_usage.get(s, 0) + 1

    overused = [s for s, count in script_usage.items() if count >= 3]

    _blocked = str(overused) if overused else "none"
    _completed_json = json.dumps(completed[-5:], indent=2)
    _scripts = str(available_scripts)
    _budget = str(state['budget_remaining'])
    _playbook_tail = playbook[-1500:]

    prompt = (
        "You are the Strategist for ABPT — Anchor-Based Probing Tool research on Qwen3.5-4B.\n\n"
        "## CONFIRMED FINDINGS (do NOT re-test)\n"
        "- tail_retention_ratio predicts constraint_score rho=+0.642 (CONFIRMED)\n"
        "- early_slope_4_8 does NOT predict (rho=-0.14, NEGATIVE)\n"
        "- Crystallization zone: L4-L8\n"
        "- Carryover layers: json=L11, contradiction=L25, DI=L24, binary=L10, vegan=L11\n\n"
        "## BLOCKED SCRIPTS (used 3+ times, do NOT propose again)\n"
        + _blocked + "\n\n"
        "## Open questions (pick ONE)\n"
        "1. Group-specific routing vs universal threshold → run_qwen_anchor_carryover_probe.py\n"
        "2. Attention beacon: does attention mass peak at L4-L8? → run_qwen_anchor_geometry_probe.py\n"
        "3. Rescue rate on flat cases → run_qwen_geometry_generation_calibration.py\n"
        "4. Write a NEW script to test tail_retention_ratio as routing gate\n\n"
        "## Completed experiments (last 5 of " + str(len(completed)) + ")\n"
        + _completed_json + "\n\n"
        "## Available scripts\n" + _scripts + "\n\n"
        "## Budget: " + _budget + " experiments remaining\n\n"
        "## Script API (if writing new script)\n"
        "- overlay = QwenAnchorOverlay(model); overlay.load()\n"
        "- cases = make_qwen_anchor_geometry_cases(anchor_profile='medium')\n"
        "- enc = encode_focus_span(overlay, case)  # returns SpanEncoding\n"
        "- enc.hidden_states[layer+1][0] -> tensor [seq_len, hidden_dim]\n"
        "- Save to: ROOT / 'archive' / 'my_output.json'\n"
        "- Imports: from src.model.qwen_anchor_overlay import QwenAnchorOverlay\n"
        "- from src.data.qwen_anchor_geometry_cases import make_qwen_anchor_geometry_cases\n"
        "- from src.utils.qwen_anchor_cartography import encode_focus_span, build_group_concept_vectors\n"
        "- from src.utils.anchor_geometry import list_model_layers\n"
        "- from src.model.config import TOY_CONFIG\n\n"
        "Respond with ONLY valid JSON (no markdown, no comments):\n"
        '{"id": "...", "description": "...", "script": "run_qwen_....py", '
        '"args": {"anchor_profile": "medium"}, "result_key": "...", '
        '"success_threshold": 0.4, "reasoning": "...", '
        '"script_code": "# omit this field if using existing script"}'
    )

    def _call_llm(prompt: str) -> str | None:
        # DeepSeek (приоритет — самый дешёвый)
        if deepseek_key:
            try:
                import openai as _openai
                client = _openai.OpenAI(api_key=deepseek_key,
                                        base_url="https://api.deepseek.com/v1")
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Strategist/DeepSeek] error: {e}")

        # Anthropic
        if anthropic_key:
            try:
                import anthropic as _anthropic
                client = _anthropic.Anthropic(api_key=anthropic_key)
                resp = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            except Exception as e:
                print(f"[Strategist/Claude] error: {e}")

        # OpenAI
        if openai_key:
            try:
                import openai as _openai
                client = _openai.OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Strategist/GPT] error: {e}")

        return None

    raw = _call_llm(prompt)
    if not raw:
        return None

    # Парсим JSON из ответа
    try:
        # Убираем возможные markdown блоки и комментарии
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        # Убираем строки-комментарии (#) перед JSON
        lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
        raw = "\n".join(lines).strip()
        # Если ответ не начинается с { — модель ответила не JSON, логируем
        if not raw.startswith("{"):
            print(f"[Strategist/LLM] Не JSON ответ: {raw[:100]!r}")
            return None
        proposal = json.loads(raw.strip())
        print(f"[Strategist/LLM] Предложение: {proposal.get('id')} — {proposal.get('description', '')[:60]}")
        print(f"[Strategist/LLM] Обоснование: {proposal.get('reasoning', '')[:120]}")
        return proposal
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Strategist/LLM] Ошибка парсинга: {e}\nRaw: {raw[:200]}")
        return None
    except Exception as e:
        print(f"[Strategist/LLM] Ошибка: {e} → fallback на rule-based")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker — запускает скрипт
# ─────────────────────────────────────────────────────────────────────────────

def worker_run(
    hyp_id: str,
    state: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    hyp_def = EXPERIMENT_REGISTRY[hyp_id]
    script_path = SCRIPTS_DIR / hyp_def["script"]

    if not script_path.exists():
        print(f"[Worker] SKIP — скрипт не найден: {script_path}")
        return {"status": "skipped", "reason": f"script not found: {hyp_def['script']}"}

    model = state["known_facts"]["model"].replace("/", "_").lower().replace("-", "_").replace(".", "")
    args = [sys.executable, str(script_path)]
    for key, val in hyp_def["default_args"].items():
        args += [f"--{key.replace('_', '-')}", str(val)]
    args += ["--model", state.get("model", "Qwen/Qwen3.5-4B")]

    print(f"[Worker] Команда: {' '.join(args)}")

    if dry_run:
        print("[Worker] DRY RUN — не запускаем")
        return {"status": "dry_run", "command": " ".join(args)}

    start_time = time.time()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=3600,  # 1 час максимум
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        output = {
            "status": "success" if success else "failed",
            "returncode": result.returncode,
            "elapsed_seconds": round(elapsed),
            "stdout_tail": result.stdout[-3000:] if result.stdout else "",
            "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        }

        if not success:
            print(f"[Worker] FAILED (code {result.returncode})")
            print(result.stderr[-500:])
        else:
            print(f"[Worker] OK в {elapsed:.0f}с")

        return output

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_seconds": 3600}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer — читает результат и обновляет state + playbook
# ─────────────────────────────────────────────────────────────────────────────

def _nested_get(d: dict, key_path: str) -> Any:
    """Получить вложенное значение по пути 'a.b.c'."""
    parts = key_path.split(".")
    val = d
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def analyzer_parse_result(hyp_id: str, worker_output: dict[str, Any]) -> dict[str, Any]:
    """Найти и распарсить JSON результат из archive/."""
    hyp_def = EXPERIMENT_REGISTRY[hyp_id]
    result_key = hyp_def.get("result_key")

    # Ищем свежие JSON файлы в archive
    pattern = hyp_def["output_pattern"].replace("archive/", "")
    matching = list(ARCHIVE_DIR.glob(pattern))
    if not matching:
        # Попробуем все JSON файлы, отсортируем по времени
        matching = sorted(ARCHIVE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not matching:
        return {"metric_value": None, "note": "no output file found"}

    latest = sorted(matching, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f"[Analyzer] Читаю: {latest.name}")

    try:
        data = json.loads(latest.read_text())
    except Exception as e:
        return {"metric_value": None, "note": f"JSON parse error: {e}"}

    metric_value = _nested_get(data, result_key) if result_key else None

    # Собираем краткий summary
    summary: dict[str, Any] = {
        "output_file": latest.name,
        "metric_key": result_key,
        "metric_value": metric_value,
    }

    # Специфичный парсинг для phase_probe
    if "phase_probe" in latest.name:
        corr = data.get("correlation_summary", {})
        all_metrics = corr.get("all_metrics", {})
        summary["correlations"] = all_metrics
        summary["n_cases"] = data.get("metadata", {}).get("n_cases")
        # Лучшая метрика
        best_metric = max(
            ((k, abs(v)) for k, v in all_metrics.items() if v is not None),
            key=lambda x: x[1],
            default=(None, None),
        )
        summary["best_predictor"] = best_metric[0]
        summary["best_rho"] = best_metric[1]
        print(f"[Analyzer] Лучший предиктор: {best_metric[0]} (|ρ|={best_metric[1]:.3f})" if best_metric[0] else "")

    return summary


def analyzer_update_playbook(hyp_id: str, analysis: dict[str, Any], state: dict[str, Any]) -> None:
    """Добавить новую запись в playbook.md."""
    hyp_def = EXPERIMENT_REGISTRY[hyp_id]
    metric_val = analysis.get("metric_value")
    success_threshold = hyp_def.get("success_threshold")

    if metric_val is not None and success_threshold is not None:
        confirmed = abs(float(metric_val)) >= float(success_threshold)
        verdict = "✅ CONFIRMED" if confirmed else "❌ NOT CONFIRMED"
    else:
        verdict = "📊 DATA COLLECTED"

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")

    new_entry = f"""
---
## [{timestamp}] Эксперимент {hyp_id}: {hyp_def['description']}

**Статус:** {verdict}  
**Метрика:** `{hyp_def.get('result_key', '—')}` = `{metric_val}`  
"""

    # Добавляем корреляции если есть
    if "correlations" in analysis:
        new_entry += "\n**Корреляции (Spearman ρ):**\n"
        for metric, rho in analysis["correlations"].items():
            if rho is not None:
                new_entry += f"- `{metric}`: {rho:.4f}\n"

    if analysis.get("best_predictor"):
        new_entry += f"\n**Лучший предиктор:** `{analysis['best_predictor']}` (|ρ|={analysis.get('best_rho', 0):.3f})\n"

    playbook_text = PLAYBOOK_FILE.read_text()
    playbook_text += new_entry
    PLAYBOOK_FILE.write_text(playbook_text)
    print(f"[Analyzer] Playbook обновлён ({verdict})")


def analyzer_update_state(
    hyp_id: str,
    analysis: dict[str, Any],
    worker_output: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Обновить research_state.json после эксперимента."""
    hyp_def = EXPERIMENT_REGISTRY[hyp_id]
    metric_val = analysis.get("metric_value")
    success_threshold = hyp_def.get("success_threshold")

    if metric_val is not None and success_threshold is not None:
        exp_status = "success" if abs(float(metric_val)) >= float(success_threshold) else "failed"
    elif worker_output.get("status") == "success":
        exp_status = "done"
    else:
        exp_status = worker_output.get("status", "unknown")

    experiment_record = {
        "hypothesis_id": hyp_id,
        "description": hyp_def["description"],
        "script": hyp_def["script"],
        "status": exp_status,
        "metric_key": hyp_def.get("result_key"),
        "metric_value": metric_val,
        "elapsed_seconds": worker_output.get("elapsed_seconds"),
        "output_file": analysis.get("output_file"),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    phase_key = str(hyp_def["phase"])
    if phase_key not in state["phases"]:
        state["phases"][phase_key] = {"experiments": []}
    state["phases"][phase_key].setdefault("experiments", []).append(experiment_record)

    # Обновить статус гипотезы в open_hypotheses
    for hyp in state.get("open_hypotheses", []):
        if hyp.get("id") == hyp_id:
            hyp["status"] = exp_status

    # Обновить metric_history
    state.setdefault("metric_history", []).append({
        "experiment": hyp_id,
        "metric": hyp_def.get("result_key"),
        "value": metric_val,
        "timestamp": datetime.now(UTC).isoformat(),
    })

    # Уменьшить бюджет
    state["budget_remaining"] = max(0, state["budget_remaining"] - 1)
    state["experiments_run"] = state.get("experiments_run", 0) + 1
    state["last_updated"] = datetime.now(UTC).strftime("%Y-%m-%d")

    # Если фаза 1 успешно завершена — разблокировать фазы 2 и 3
    phase1_exps = state["phases"].get("1", {}).get("experiments", [])
    phase1_success = any(e["status"] == "success" for e in phase1_exps)
    if phase1_success:
        for blocked_phase in ["2", "3"]:
            if state["phases"].get(blocked_phase, {}).get("status") == "blocked_on_phase_1":
                state["phases"][blocked_phase]["status"] = "pending"
                print(f"[Analyzer] Фаза {blocked_phase} разблокирована")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def print_status(state: dict[str, Any]) -> None:
    print(f"\n{SEPARATOR}")
    print(f"📊 ABPT Research State")
    print(f"   Phase:    {state['current_phase']}/6")
    print(f"   Budget:   {state['budget_remaining']}/{state['experiments_max']}")
    print(f"   Runs:     {state.get('experiments_run', 0)}")

    completed = []
    for phase_data in state["phases"].values():
        for exp in phase_data.get("experiments", []):
            completed.append(f"{exp['hypothesis_id']}={exp['status']}")
    if completed:
        print(f"   Done:     {', '.join(completed)}")

    last = state.get("metric_history", [])[-1] if state.get("metric_history") else None
    if last:
        print(f"   Last metric: {last['metric']} = {last['value']}")
    print(SEPARATOR)


def run_loop(
    budget: int | None = None,
    target_phase: int | None = None,
    target_hypothesis: str | None = None,
    dry_run: bool = False,
    use_llm_strategist: bool = False,
) -> None:
    state = load_state()
    if budget is not None:
        state["budget_remaining"] = budget  # override: --budget always wins

    # Добавляем поле model в known_facts если нет
    if "model" not in state["known_facts"]:
        state["known_facts"]["model"] = state.get("model", "Qwen/Qwen3.5-4B")

    playbook = PLAYBOOK_FILE.read_text() if PLAYBOOK_FILE.exists() else ""

    print(f"\n{'═'*60}")
    print(f"🔬 ABPT Orchestrator запущен")
    print(f"   Model: {state.get('model', 'Qwen/Qwen3.5-4B')}")
    print(f"   Budget: {state['budget_remaining']}")
    print(f"   DRY RUN: {dry_run}")
    print(f"{'═'*60}")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n[Loop {iteration}] Strategist выбирает следующий эксперимент...")

        # Выбор следующего эксперимента
        llm_proposal: dict[str, Any] | None = None
        if target_hypothesis:
            hyp_id = target_hypothesis
            target_hypothesis = None
        elif use_llm_strategist:
            llm_proposal = strategist_llm_select(state, playbook)
            if llm_proposal:
                hyp_id = llm_proposal["id"]
                # Регистрируем динамически в EXPERIMENT_REGISTRY
                script_name = llm_proposal.get("script", f"run_qwen_llm_{hyp_id}.py")
                # Если DeepSeek написал код — сохраняем скрипт
                script_code = llm_proposal.get("script_code", "")
                if script_code and script_code.strip():
                    script_path = SCRIPTS_DIR / script_name
                    script_path.write_text(script_code)
                    print(f"[Strategist/LLM] Написан новый скрипт: {script_name} ({len(script_code)} chars)")

                EXPERIMENT_REGISTRY[hyp_id] = {
                    "description": llm_proposal.get("description", hyp_id),
                    "phase": state["current_phase"],
                    "script": script_name,
                    "default_args": llm_proposal.get("args", {}),
                    "output_pattern": "archive/*.json",
                    "result_key": llm_proposal.get("result_key"),
                    "success_threshold": llm_proposal.get("success_threshold"),
                    "depends_on": [],
                }
            else:
                hyp_id = strategist_select_next(state, target_phase)
        else:
            hyp_id = strategist_select_next(state, target_phase)

        if hyp_id is None:
            print("\n[Orchestrator] Нет доступных экспериментов. Кампания завершена.")
            break

        hyp_def = EXPERIMENT_REGISTRY[hyp_id]
        source = "LLM" if llm_proposal else "rule-based"
        print(f"[Strategist/{source}] → {hyp_id}: {hyp_def['description']}")
        if llm_proposal and llm_proposal.get("reasoning"):
            print(f"[Strategist/reasoning] {llm_proposal['reasoning'][:150]}")
        print_status(state)

        log_event({"event": "experiment_start", "hypothesis": hyp_id,
                   "llm_proposal": llm_proposal})

        # Worker запускает скрипт
        print(f"\n[Worker] Запускаю {hyp_def['script']}...")
        worker_output = worker_run(hyp_id, state, dry_run=dry_run)

        if worker_output["status"] == "dry_run":
            print("[Orchestrator] DRY RUN завершён.")
            break

        if worker_output["status"] == "skipped":
            print(f"[Orchestrator] SKIP {hyp_id} — скрипт не существует ещё")
            # Помечаем как skipped и продолжаем
            state.setdefault("phases", {}).setdefault(str(hyp_def["phase"]), {}).setdefault("experiments", []).append({
                "hypothesis_id": hyp_id, "status": "skipped",
                "timestamp": datetime.now(UTC).isoformat(),
            })
            state["budget_remaining"] = max(0, state["budget_remaining"] - 1)
            save_state(state)
            continue

        # Analyzer разбирает результат
        print(f"\n[Analyzer] Разбираю результат...")
        analysis = analyzer_parse_result(hyp_id, worker_output)
        analyzer_update_playbook(hyp_id, analysis, state)
        analyzer_update_state(hyp_id, analysis, worker_output, state)

        playbook = PLAYBOOK_FILE.read_text()  # обновить после записи

        log_event({
            "event": "experiment_done",
            "hypothesis": hyp_id,
            "status": analysis,
            "metric": analysis.get("metric_value"),
        })

        save_state(state)

        # Стоп если бюджет исчерпан
        if state["budget_remaining"] <= 0:
            print("\n[Orchestrator] Бюджет исчерпан.")
            break

        print(f"\n[Orchestrator] Пауза 3с перед следующим экспериментом...")
        time.sleep(3)

    print(f"\n{'═'*60}")
    print(f"🏁 Кампания завершена")
    print(f"   Запущено экспериментов: {state.get('experiments_run', 0)}")
    print(f"   Бюджет остаток: {state['budget_remaining']}")
    print(f"   Лог: {LOG_FILE}")
    print(f"   State: {STATE_FILE}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ABPT Research Orchestrator")
    parser.add_argument("--budget", type=int, default=None,
                        help="Максимум экспериментов в этой сессии")
    parser.add_argument("--phase", type=int, default=None,
                        help="Запустить только эксперименты указанной фазы")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Запустить конкретную гипотезу (H1, H2, ...)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать план без запуска")
    parser.add_argument("--llm-strategist", action="store_true",
                        help="Использовать LLM для выбора экспериментов (нужен API ключ)")
    parser.add_argument("--status", action="store_true",
                        help="Показать текущий state и выйти")
    args = parser.parse_args()

    if args.status:
        state = load_state()
        print_status(state)
        return

    run_loop(
        budget=args.budget,
        target_phase=args.phase,
        target_hypothesis=args.experiment,
        dry_run=args.dry_run,
        use_llm_strategist=args.llm_strategist,
    )


if __name__ == "__main__":
    main()
