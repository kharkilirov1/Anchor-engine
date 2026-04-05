"""
Local Strategist — CLI AI-стратег с backend Claude или Codex.

Полностью свободный research agent: сам читает проект, сам решает
какой эксперимент запускать, может предложить новые гипотезы и написать
новые скрипты.

Использование:
    python scripts/orchestrate.py --local-strategist --max-turns 10
    python scripts/orchestrate.py --local-strategist --hours 8 --budget 20
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAX_TURNS = 10  # больше turns = глубже reasoning
DEFAULT_PER_CALL_TIMEOUT = 600  # 10 минут на один вызов
DEFAULT_BACKEND = "codex"
DEFAULT_CODEX_MODE = "autonomous"
DEFAULT_FALLBACK_CHAIN = "codex,claude"


def _list_run_qwen_scripts() -> list[str]:
    return sorted(path.name for path in (ROOT / "scripts").glob("run_qwen_*.py"))


def _build_codex_template_catalog() -> list[dict[str, Any]]:
    available = set(_list_run_qwen_scripts())
    templates = [
        {
            "template_id": "phase_probe_short",
            "description": "Check whether tail_retention_ratio remains predictive on short anchor spans.",
            "script": "run_qwen_phase_probe.py",
            "args": {"anchor_profile": "short", "tau": "0.5"},
            "result_key": "correlation_summary.all_metrics.tail_retention_ratio",
            "success_threshold": 0.4,
        },
        {
            "template_id": "phase_probe_long",
            "description": "Check whether tail_retention_ratio remains predictive on long anchor spans.",
            "script": "run_qwen_phase_probe.py",
            "args": {"anchor_profile": "long", "tau": "0.5"},
            "result_key": "correlation_summary.all_metrics.tail_retention_ratio",
            "success_threshold": 0.4,
        },
        {
            "template_id": "cross_profile_probe",
            "description": "Run one robust cross-profile probe over short, medium, and long spans.",
            "script": "run_qwen_cross_profile_probe.py",
            "args": {},
            "result_key": "cross_profile.medium.tail_retention_rho",
            "success_threshold": 0.4,
        },
        {
            "template_id": "per_case_diag_short",
            "description": "Run parameterized per-case diagnostic on short profile to validate correlation directly.",
            "script": "run_qwen_per_case_diagnostic_v2.py",
            "args": {"profile": "short"},
            "result_key": "spearman_rho",
            "success_threshold": 0.4,
        },
        {
            "template_id": "per_case_diag_long",
            "description": "Run parameterized per-case diagnostic on long profile to validate correlation directly.",
            "script": "run_qwen_per_case_diagnostic_v2.py",
            "args": {"profile": "long"},
            "result_key": "spearman_rho",
            "success_threshold": 0.4,
        },
        {
            "template_id": "geometry_probe",
            "description": "Strengthen or falsify the beacon/crystallization interpretation with tokenization-controlled geometry probe.",
            "script": "run_qwen_anchor_geometry_probe.py",
            "args": {},
            "result_key": "interpretation.support_after_tokenization_controls",
            "success_threshold": None,
        },
        {
            "template_id": "carryover_medium",
            "description": "Measure whether group-specific carryover beats a universal threshold on medium profile.",
            "script": "run_qwen_anchor_carryover_probe.py",
            "args": {"profiles": ["medium"]},
            "result_key": "summary.mean_last_token_delta",
            "success_threshold": 0.02,
        },
        {
            "template_id": "layer_profile_map",
            "description": "Map peak geometry layers across profiles to refine the crystallization zone hypothesis.",
            "script": "run_qwen_anchor_layer_profile_map.py",
            "args": {},
            "result_key": "summary.trimmed_rank1_peak_layer_mean",
            "success_threshold": None,
        },
    ]
    return [template for template in templates if template["script"] in available]


def _build_codex_template_prompt(state: dict[str, Any], playbook: str) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    catalog = _build_codex_template_catalog()
    catalog_by_id = {item["template_id"]: item for item in catalog}
    completed = []
    script_counts: dict[str, int] = {}
    for phase_data in state.get("phases", {}).values():
        for exp in phase_data.get("experiments", []):
            completed.append(
                {
                    "id": exp.get("hypothesis_id"),
                    "status": exp.get("status"),
                    "metric": exp.get("metric_value"),
                    "script": exp.get("script"),
                }
            )
            script_name = str(exp.get("script", ""))
            if script_name:
                script_counts[script_name] = script_counts.get(script_name, 0) + 1

    catalog_lines = []
    for item in catalog:
        catalog_lines.append(
            f"- {item['template_id']}: script={item['script']} "
            f"args={json.dumps(item['args'], ensure_ascii=False)} "
            f"metric={item['result_key']} "
            f"threshold={item['success_threshold']} "
            f"goal={item['description']}"
        )

    recent_history = json.dumps(completed[-12:], indent=2, ensure_ascii=False)
    known_facts = json.dumps(state.get("known_facts", {}), indent=2, ensure_ascii=False)
    blocked = [name for name, count in sorted(script_counts.items()) if count >= 3]
    prompt = (
        "You are selecting the next ABPT experiment template for a Codex-driven orchestrator.\n\n"
        "Hard rules:\n"
        "- Choose EXACTLY ONE template_id from the catalog below.\n"
        "- Do not answer with 'insufficient data'. Pick the best available template.\n"
        "- Prefer experiments that resolve current uncertainty, validate suspicious null results, or avoid known broken paths.\n"
        "- Do not invent script names or CLI flags.\n"
        "- Keep reasoning short and concrete.\n\n"
        f"Current phase: {state.get('current_phase')}\n"
        f"Budget remaining: {state.get('budget_remaining')}\n"
        f"Known facts:\n{known_facts}\n\n"
        f"Recently completed experiments:\n{recent_history}\n\n"
        f"Overused script families (3+ runs): {blocked or 'none'}\n\n"
        "Catalog:\n"
        + "\n".join(catalog_lines)
        + "\n\n"
        "Playbook tail:\n"
        + playbook[-1200:]
        + "\n\nReturn only JSON matching the schema."
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "template_id": {"type": "string", "enum": list(catalog_by_id.keys())},
            "reasoning": {"type": "string"},
        },
        "required": ["template_id", "reasoning"],
        "additionalProperties": False,
    }
    return prompt, schema, catalog_by_id


def _build_codex_free_existing_prompt(state: dict[str, Any], playbook: str) -> tuple[str, dict[str, Any]]:
    available_scripts = _list_run_qwen_scripts()
    cpu_mode = os.environ.get("ALLOW_CPU_SPACE", "").strip() == "1"
    script_help = {
        "run_qwen_phase_probe.py": "tail_retention / phase metrics correlation probe; args like anchor_profile, tau",
        "run_qwen_cross_profile_probe.py": "robust short/medium/long correlation comparison; no extra args needed",
        "run_qwen_per_case_diagnostic_v2.py": "flat per-case diagnostic; args like profile=short|medium|long",
        "run_qwen_anchor_geometry_probe.py": "tokenization-controlled geometry/beacon evidence probe",
        "run_qwen_anchor_carryover_probe.py": "carryover vs routing threshold comparison; args like profiles=['medium']",
        "run_qwen_geometry_generation_calibration.py": "generation calibration / rescue evaluation on geometry-selected cases",
        "run_qwen_anchor_layer_profile_map.py": "map rank1/coherence peak layers across profiles",
        "run_qwen_anchor_geometry_profile_sweep.py": "profile sweep over geometry support summaries",
        "run_qwen_anchor_length_sweep.py": "length sweep and flat-cluster behavior",
        "run_qwen_future_influence_probe.py": "future-influence summary and conflict-vs-stable gap metrics",
        "run_qwen_anchor_concept_direction_map.py": "concept-vector direction maps by group",
        "run_qwen_long_retention_compare.py": "long retention compare on one prompt",
    }
    lines = []
    for script in available_scripts:
        lines.append(f"- {script}: {script_help.get(script, 'existing runnable script in repo')}")

    completed = []
    for phase_data in state.get("phases", {}).values():
        for exp in phase_data.get("experiments", []):
            completed.append(
                {
                    "id": exp.get("hypothesis_id"),
                    "status": exp.get("status"),
                    "metric": exp.get("metric_value"),
                    "script": exp.get("script"),
                }
            )

    cpu_hint = (
        "\nCPU-only overnight mode is active:\n"
        "- Prefer `run_qwen_per_case_diagnostic_v2.py`\n"
        "- Use args like {\"profile\":\"short|medium|long\",\"max_new_tokens\":8,\"group_case_cap\":1,\"device\":\"cpu\"}\n"
        "- Avoid heavy multi-profile or long-generation probes unless explicitly necessary\n\n"
        if cpu_mode
        else ""
    )

    prompt = (
        "You are the ABPT Codex strategist. "
        "Choose the next experiment using ONLY an existing script from the allowed list.\n\n"
        "Hard rules:\n"
        "- script must be one of the allowed existing scripts below\n"
        "- args_json must be a valid JSON object encoded as a string\n"
        "- do not invent scripts or unsupported CLI flags\n"
        "- prioritize experiments that resolve current uncertainty or avoid known broken paths\n"
        "- output only JSON matching the schema\n\n"
        f"Current phase: {state.get('current_phase')}\n"
        f"Budget remaining: {state.get('budget_remaining')}\n"
        f"Known facts: {json.dumps(state.get('known_facts', {}), ensure_ascii=False)}\n"
        f"Recent experiments: {json.dumps(completed[-12:], ensure_ascii=False)}\n\n"
        + cpu_hint +
        "Allowed scripts:\n"
        + "\n".join(lines)
        + "\n\nPlaybook tail:\n"
        + playbook[-1200:]
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "description": {"type": "string"},
            "script": {"type": "string", "enum": available_scripts},
            "args_json": {"type": "string"},
            "result_key": {"type": "string"},
            "success_threshold": {"type": ["number", "null"]},
            "reasoning": {"type": "string"},
            "script_code": {"type": "string"},
        },
        "required": [
            "id",
            "description",
            "script",
            "args_json",
            "result_key",
            "success_threshold",
            "reasoning",
            "script_code",
        ],
        "additionalProperties": False,
    }
    return prompt, schema


def _load_recent_failure_context(limit: int = 8) -> list[dict[str, Any]]:
    log_path = ROOT / "orchestrator_log.jsonl"
    if not log_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "experiment_done":
                continue
            status = payload.get("status", {})
            metric = status.get("metric_value") if isinstance(status, dict) else None
            note = status.get("note") if isinstance(status, dict) else None
            if metric is None or note:
                rows.append(
                    {
                        "hypothesis": payload.get("hypothesis"),
                        "metric_value": metric,
                        "note": note,
                        "timestamp": payload.get("timestamp"),
                    }
                )
    except Exception:
        return []
    return rows[-limit:]


def _build_codex_autonomous_prompt(state: dict[str, Any], playbook: str) -> tuple[str, dict[str, Any]]:
    available_scripts = _list_run_qwen_scripts()
    recent_failures = _load_recent_failure_context()
    cpu_mode = os.environ.get("ALLOW_CPU_SPACE", "").strip() == "1"
    recent_history = []
    for phase_data in state.get("phases", {}).values():
        for exp in phase_data.get("experiments", []):
            recent_history.append(
                {
                    "id": exp.get("hypothesis_id"),
                    "status": exp.get("status"),
                    "metric": exp.get("metric_value"),
                    "script": exp.get("script"),
                }
            )

    cpu_hint = (
        "CPU-only overnight mode is active. Treat the HF Space as CPU hardware.\n"
        "Prefer lightweight diagnostics, especially `run_qwen_per_case_diagnostic_v2.py` with "
        "`max_new_tokens<=8`, `group_case_cap=1`, `device=cpu`, and one profile at a time.\n"
        "Do not select expensive multi-profile probes unless you have already verified they fit.\n\n"
        if cpu_mode
        else ""
    )

    prompt = (
        "You are the autonomous overnight research strategist/operator for the ABPT project.\n\n"
        "Goal: behave like a real autonomous researcher. Inspect the repository, understand the current state, "
        "repair blocking pipeline problems when needed, and then choose the best next experiment.\n\n"
        "You are allowed to:\n"
        "- inspect and modify files inside this repo\n"
        "- run Python scripts, py_compile, targeted pytest, and other safe diagnostics\n"
        "- repair broken experiment scripts, orchestrator glue, or worker integration if that is the highest-value blocker\n"
        "- add a new experiment script under scripts/ when necessary\n\n"
        "You must:\n"
        "1. read the project state from research_state.json and playbook.md\n"
        "2. inspect relevant scripts/tests/logs instead of guessing\n"
        "3. if the pipeline is blocked by a concrete bug, fix it first and verify with the lightest convincing check\n"
        "4. then return ONE next experiment proposal as JSON\n"
        "5. prefer proposals that can actually run tonight on the existing HF worker / local pipeline\n\n"
        "Hard constraints for the final JSON:\n"
        f"- script must end with .py and should normally live under scripts/\n"
        f"- if using an existing script, prefer one of these known runnable scripts: {available_scripts}\n"
        "- args_json must be a JSON object encoded as a string\n"
        "- success_threshold may be null for qualitative experiments\n"
        "- script_code may be empty if you already modified/created files directly in the repo\n"
        "- output only JSON in the final answer\n\n"
        f"Current phase: {state.get('current_phase')}\n"
        f"Budget remaining: {state.get('budget_remaining')}\n"
        f"Known facts: {json.dumps(state.get('known_facts', {}), ensure_ascii=False)}\n"
        f"Recent experiments: {json.dumps(recent_history[-12:], ensure_ascii=False)}\n"
        f"Recent failures / null results: {json.dumps(recent_failures, ensure_ascii=False)}\n\n"
        + cpu_hint +
        "Important repo locations to inspect:\n"
        "- research_state.json\n"
        "- playbook.md\n"
        "- scripts/\n"
        "- tests/\n"
        "- archive/\n"
        "- docs/research/\n\n"
        "Current playbook tail:\n"
        + playbook[-2000:]
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "description": {"type": "string"},
            "script": {"type": "string"},
            "args_json": {"type": "string"},
            "result_key": {"type": "string"},
            "success_threshold": {"type": ["number", "null"]},
            "reasoning": {"type": "string"},
            "script_code": {"type": "string"},
        },
        "required": [
            "id",
            "description",
            "script",
            "args_json",
            "result_key",
            "success_threshold",
            "reasoning",
            "script_code",
        ],
        "additionalProperties": False,
    }
    return prompt, schema


def strategist_local_select(
    state: dict[str, Any],
    playbook: str,
    *,
    max_turns: int | None = None,
    per_call_timeout: int | None = None,
) -> dict[str, Any] | None:
    """
    CLI Strategist — полностью автономный research agent.
    Читает проект, анализирует состояние, предлагает следующий эксперимент.
    """
    _max_turns = max_turns or int(os.environ.get("STRATEGIST_MAX_TURNS", DEFAULT_MAX_TURNS))
    _timeout = per_call_timeout or int(os.environ.get("STRATEGIST_TIMEOUT", DEFAULT_PER_CALL_TIMEOUT))
    preferred_backend = os.environ.get("STRATEGIST_BACKEND", DEFAULT_BACKEND)
    codex_mode = os.environ.get("STRATEGIST_CODEX_MODE", DEFAULT_CODEX_MODE)
    backend_order = _build_backend_order(preferred_backend)

    print(
        "[Strategist/Local] "
        f"preferred_backend={preferred_backend}, "
        f"fallback_chain={backend_order}, "
        f"codex_mode={codex_mode}, "
        f"max_turns={_max_turns}, timeout={_timeout}s"
    )

    last_raw: str | None = None
    proposal: dict[str, Any] | None = None
    for backend in backend_order:
        if backend == "codex":
            proposal, last_raw = _run_codex_backend(
                state,
                playbook,
                timeout=_timeout,
                codex_mode=codex_mode,
            )
        elif backend == "claude":
            proposal, last_raw = _run_claude_backend(
                state,
                max_turns=_max_turns,
                timeout=_timeout,
            )
        else:
            print(f"[Strategist/Local] Skip unknown backend in chain: {backend}")
            continue
        if proposal is not None:
            if backend != preferred_backend:
                print(f"[Strategist/Local] Fallback backend succeeded: {backend}")
            break
        print(f"[Strategist/Local] Backend {backend} did not produce valid proposal")

    if not last_raw:
        print("[Strategist/Local] No response from any backend -> fallback to rule-based")
        return None

    if proposal:
        print(f"[Strategist/Local] Proposal: {proposal.get('id')} — {proposal.get('description', '')[:80]}")
        reasoning = proposal.get('reasoning', '')
        if reasoning:
            print(f"[Strategist/Local] Reasoning: {reasoning[:200]}")

    return proposal


def _build_backend_order(preferred_backend: str) -> list[str]:
    raw_chain = os.environ.get("STRATEGIST_FALLBACK_CHAIN", DEFAULT_FALLBACK_CHAIN)
    requested = [item.strip().lower() for item in raw_chain.split(",") if item.strip()]
    known = [item for item in requested if item in {"codex", "claude"}]
    if preferred_backend in {"codex", "claude"}:
        ordered = [preferred_backend]
        ordered.extend(item for item in known if item != preferred_backend)
        alternate = "claude" if preferred_backend == "codex" else "codex"
        if alternate not in ordered:
            ordered.append(alternate)
        return ordered
    return known or [DEFAULT_BACKEND, "claude"]


def _run_claude_backend(
    state: dict[str, Any],
    *,
    max_turns: int,
    timeout: int,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt = _build_free_strategist_prompt(state)
    raw = _call_claude_code(prompt, max_turns, timeout)
    proposal = _parse_strategist_response(raw) if raw else None
    return proposal, raw


def _run_codex_backend(
    state: dict[str, Any],
    playbook: str,
    *,
    timeout: int,
    codex_mode: str,
) -> tuple[dict[str, Any] | None, str | None]:
    last_raw: str | None = None
    proposal: dict[str, Any] | None = None

    if codex_mode == "autonomous":
        prompt, schema = _build_codex_autonomous_prompt(state, playbook)
        last_raw = _call_codex(prompt, timeout, schema=schema, workspace_write=True)
        proposal = _parse_strategist_response(last_raw) if last_raw else None
        if proposal is not None:
            return proposal, last_raw

    prompt, schema = _build_codex_free_existing_prompt(state, playbook)
    last_raw = _call_codex(prompt, timeout, schema=schema)
    proposal = _parse_strategist_response(last_raw) if last_raw else None
    if proposal is not None:
        return proposal, last_raw

    fallback_prompt, fallback_schema, catalog_by_id = _build_codex_template_prompt(state, playbook)
    last_raw = _call_codex(fallback_prompt, timeout, schema=fallback_schema)
    selection = _parse_json_from_mixed_output(last_raw) if last_raw else None
    if isinstance(selection, dict):
        template_id = str(selection.get("template_id", "")).strip()
        template = catalog_by_id.get(template_id)
        if template:
            proposal = {
                "id": f"{template_id}_{int(time.time())}",
                "description": template["description"],
                "script": template["script"],
                "args": dict(template["args"]),
                "result_key": template["result_key"],
                "success_threshold": template["success_threshold"],
                "reasoning": str(selection.get("reasoning", "")).strip(),
                "script_code": "",
            }
    return proposal, last_raw


# ─────────────────────────────────────────────────────────────────────────────
# Free-form Strategist Prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_free_strategist_prompt(state: dict[str, Any]) -> str:
    """
    Промпт без ограничений — стратег сам исследует проект и решает.
    """
    budget = state.get("budget_remaining", 0)
    phase = state.get("current_phase", 1)

    # Собираем историю экспериментов
    experiments_history = []
    for phase_key, phase_data in state.get("phases", {}).items():
        for exp in phase_data.get("experiments", []):
            experiments_history.append(
                f"  - {exp['hypothesis_id']}: {exp['status']}"
                f" (metric={exp.get('metric_value', 'N/A')})"
            )
    history_str = "\n".join(experiments_history[-15:]) if experiments_history else "  (no experiments yet)"

    known_facts = json.dumps(state.get("known_facts", {}), indent=2, ensure_ascii=False)

    return f"""You are an autonomous research strategist for the ABPT project.

## YOUR MISSION
Decide what experiment to run next. You have FULL FREEDOM:
- You can propose ANY hypothesis, not just predefined ones
- You can reuse existing scripts with new arguments
- You can write entirely NEW experiment scripts
- You can change research direction based on findings
- You can explore novel angles nobody has considered

## HOW TO DECIDE
1. Read `research_state.json` to understand current progress
2. Read `playbook.md` to see accumulated findings
3. Look at `scripts/run_qwen_*.py` to see available experiments
4. Look at `archive/` for past results
5. Read `src/data/` to understand available test cases
6. Think about what experiment would produce the MOST VALUABLE new knowledge

## CURRENT STATE
- Phase: {phase}
- Budget remaining: {budget} experiments
- Known facts: {known_facts}

## EXPERIMENT HISTORY (last 15)
{history_str}

## AVAILABLE SCRIPTS (in scripts/ directory)
Read them yourself to understand their arguments and capabilities.

## CONSTRAINTS
- Each experiment runs as: `python scripts/<script>.py --model-name Qwen/Qwen3.5-4B [args]`
- Scripts run on a REMOTE worker (HF Space) — they must be self-contained
- If you write a new script, put the full code in "script_code" field
- Budget is limited — choose experiments that maximize information gain

## WHAT TO CONSIDER
- What hypotheses have been confirmed vs rejected?
- What gaps remain in our understanding?
- Are there patterns in the data we haven't explored?
- Can we combine findings from multiple experiments?
- Is there a novel angle that could yield breakthrough insights?
- Should we replicate a key finding for confidence?
- Should we explore a completely new direction?

## OUTPUT
Respond with ONLY a valid JSON object (no markdown, no explanation, no commentary):
{{
  "id": "unique_experiment_id",
  "description": "What this experiment tests and why",
  "script": "run_qwen_something.py",
  "args": {{"key": "value"}},
  "args_json": "{{\"key\": \"value\"}}",
  "result_key": "path.to.metric.in.output.json",
  "success_threshold": 0.5,
  "reasoning": "Detailed explanation of WHY this experiment NOW — what knowledge gap it fills, how it builds on previous findings",
  "script_code": ""
}}

You may provide EITHER:
- "args" as a JSON object
- OR "args_json" as a JSON-encoded string

If using an existing script, leave "script_code" empty string "".
If writing a new script, put the FULL Python code in "script_code".
IMPORTANT: Prefer existing scripts with different args over writing new ones.
Only write new scripts when no existing script can test your hypothesis.
Keep new scripts under 150 lines to avoid JSON truncation.
The "id" must be unique — not reuse any from experiment history.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Claude Code CLI
# ─────────────────────────────────────────────────────────────────────────────

def _call_claude_code(prompt: str, max_turns: int, timeout: int) -> str | None:
    """Вызывает `claude -p` (print mode, non-interactive)."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        print("[Strategist/Local] ERROR: `claude` CLI not found in PATH")
        return None

    cmd = [
        claude_bin,
        "-p",
        "--output-format", "text",
        "--max-turns", str(max_turns),
        prompt,
    ]

    print(f"[Strategist/Local] Calling claude -p (max_turns={max_turns}, timeout={timeout}s)...")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            cwd=str(ROOT),
            env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL": "1", "PYTHONIOENCODING": "utf-8"},
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[Strategist/Local] TIMEOUT after {elapsed:.0f}s")
        return None
    except FileNotFoundError:
        print("[Strategist/Local] ERROR: claude binary not found")
        return None

    elapsed = time.time() - t0
    print(f"[Strategist/Local] Claude responded in {elapsed:.1f}s (exit={result.returncode})")

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-500:]
        print(f"[Strategist/Local] claude stderr: {stderr_tail}")
        return None

    return result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Codex CLI
# ─────────────────────────────────────────────────────────────────────────────

def _call_codex(
    prompt: str,
    timeout: int,
    *,
    schema: dict[str, Any] | None = None,
    workspace_write: bool = False,
) -> str | None:
    """Вызывает OpenAI Codex CLI в non-interactive режиме."""
    codex_bin = shutil.which("codex")
    if not codex_bin:
        print("[Strategist/Local] ERROR: `codex` CLI not found in PATH")
        return None

    archive_dir = ROOT / "archive"
    archive_dir.mkdir(exist_ok=True)
    schema_path = archive_dir / "codex_strategist_schema.json"
    if schema is not None:
        schema_path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    cmd = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--color", "never",
        "--cd", str(ROOT),
    ]
    if workspace_write:
        cmd.append("--full-auto")
    else:
        cmd.extend(["--sandbox", "read-only"])
    if schema is not None:
        cmd.extend(["--output-schema", str(schema_path)])
    cmd.append(prompt)

    print(f"[Strategist/Local] Calling codex (timeout={timeout}s)...")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        print(f"[Strategist/Local] TIMEOUT after {time.time() - t0:.0f}s")
        return None
    except FileNotFoundError:
        print("[Strategist/Local] ERROR: codex binary not found")
        return None

    elapsed = time.time() - t0
    print(f"[Strategist/Local] Codex responded in {elapsed:.1f}s (exit={result.returncode})")

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-500:]
        print(f"[Strategist/Local] codex stderr: {stderr_tail}")
        return None

    raw_output = result.stdout or ""
    debug_path = archive_dir / "codex_strategist_last_stdout.txt"
    debug_path.write_text(raw_output, encoding="utf-8", errors="replace")
    return raw_output


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_from_mixed_output(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None

def _parse_strategist_response(raw: str) -> dict[str, Any] | None:
    """Парсит JSON-ответ от стратега. Robust к markdown и мусору вокруг JSON."""
    try:
        direct = _parse_json_from_mixed_output(raw)
        if direct is not None:
            proposal = direct
        else:
            raw = raw.strip()

            # Strip markdown code fences
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break

            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                print(f"[Strategist/Local] No JSON found in response: {raw[:300]!r}")
                return None

            raw = raw[start:end + 1]
            lines = []
            for line in raw.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("//"):
                    continue
                if "//" in line and '"' not in line.split("//")[-1]:
                    line = line[:line.index("//")]
                lines.append(line)
            raw = "\n".join(lines)
            proposal = json.loads(raw)

        required = ["id", "description", "script"]
        for field in required:
            if field not in proposal:
                print(f"[Strategist/Local] Missing required field: {field}")
                return None

        if "args" not in proposal:
            args_json = proposal.get("args_json", "")
            if args_json:
                try:
                    proposal["args"] = json.loads(args_json)
                except json.JSONDecodeError:
                    print(f"[Strategist/Local] Bad args_json: {args_json!r}")
                    proposal["args"] = {}
            else:
                proposal["args"] = {}

        script_name = str(proposal.get("script", "")).strip()
        script_code = str(proposal.get("script_code", "")).strip()
        if not script_name.endswith(".py"):
            print(f"[Strategist/Local] Invalid script field: {script_name!r}")
            return None
        if not (ROOT / "scripts" / script_name).exists() and not script_code:
            print(f"[Strategist/Local] Unknown script without script_code: {script_name!r}")
            return None

        return proposal

    except json.JSONDecodeError as e:
        print(f"[Strategist/Local] JSON parse error: {e}")
        print(f"[Strategist/Local] Raw (first 500 chars): {raw[:500]!r}")
        return None
    except Exception as e:
        print(f"[Strategist/Local] Parse error: {e}")
        return None


# Backward compatibility alias
strategist_llm_select = strategist_local_select
