from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.autoresearch import (
    DEFAULT_FRONTIER_MODELS,
    ExperimentSpec,
    build_command,
    build_frontier_specs,
    build_global_leaderboard,
    choose_next_experiment,
    leaderboard_markdown,
    load_benchmark_results,
    rank_candidate_experiments,
)

OPEN_MODEL_CATALOG: tuple[str, ...] = (
    "uniform",
    "motif",
    "motif_fast",
    "structured",
    "structured_runtime",
    "structured_v2",
    "structured_v2_copy",
    "structured_fast",
    "structured_copy",
    "structured_code",
    "structured_code_light",
    "structured_easy",
    "structured_fast_easy",
)
OPEN_ALLOWED_STEP_BUDGETS: tuple[int, ...] = (60, 100, 150, 200)
OPEN_ALLOWED_TIME_BUDGETS: tuple[float, ...] = (20.0, 30.0, 40.0)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(output_dir: Path) -> Path:
    return output_dir / "loop_state.json"


def _leaderboard_path(output_dir: Path) -> Path:
    return output_dir / "leaderboard.md"


def _load_state(output_dir: Path) -> dict[str, object]:
    path = _state_path(output_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "created_at": _timestamp(),
        "iterations": [],
    }


def _save_state(output_dir: Path, state: dict[str, object]) -> None:
    _state_path(output_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_leaderboard(output_dir: Path, leaderboard_rows: list[dict[str, object]]) -> None:
    _leaderboard_path(output_dir).write_text(leaderboard_markdown(leaderboard_rows), encoding="utf-8")


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(int(item) for item in values)


def _default_seed_list() -> str:
    return ",".join(str(seed) for seed in range(42, 142))


def _candidate_payload(decision: object) -> dict[str, object]:
    from src.utils.autoresearch import SelectionDecision

    assert isinstance(decision, SelectionDecision)
    spec = decision.spec
    return {
        "candidate_id": spec.slug(),
        "name": spec.name,
        "dataset": spec.dataset,
        "mode": "time" if spec.time_budget_s > 0.0 else "steps",
        "seed": spec.seed,
        "steps": spec.steps,
        "time_budget_s": spec.time_budget_s,
        "models": list(spec.models),
        "pending_models": list(decision.pending_models),
        "score": round(decision.score, 4),
        "reason": decision.reason,
    }


def _load_recent_loop_iterations(state: dict[str, object], limit: int = 8) -> list[dict[str, object]]:
    iterations = state.get("iterations", [])
    if not isinstance(iterations, list):
        return []
    trimmed: list[dict[str, object]] = []
    for item in iterations[-limit:]:
        if not isinstance(item, dict):
            continue
        trimmed.append(
            {
                "iteration": item.get("iteration"),
                "name": item.get("selected_spec", {}).get("name") if isinstance(item.get("selected_spec"), dict) else None,
                "dataset": item.get("selected_spec", {}).get("dataset") if isinstance(item.get("selected_spec"), dict) else None,
                "seed": item.get("selected_spec", {}).get("seed") if isinstance(item.get("selected_spec"), dict) else None,
                "status": item.get("status"),
                "reason": item.get("selection_reason"),
            }
        )
    return trimmed


def _candidate_schema(candidate_ids: list[str]) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "enum": candidate_ids},
            "reasoning": {"type": "string"},
            "hypothesis": {"type": "string"},
        },
        "required": ["candidate_id", "reasoning", "hypothesis"],
        "additionalProperties": False,
    }


def _open_strategy_schema(
    allowed_models: tuple[str, ...],
    allowed_seeds: tuple[int, ...],
) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "enum": ["tinystories", "code"]},
            "comparison_mode": {"type": "string", "enum": ["steps", "time"]},
            "steps": {"type": ["integer", "null"]},
            "time_budget_s": {"type": ["number", "null"]},
            "seed": {"type": "integer", "enum": list(allowed_seeds)},
            "models": {
                "type": "array",
                "items": {"type": "string", "enum": list(allowed_models)},
                "minItems": 2,
                "maxItems": 6,
                "uniqueItems": True,
            },
            "reasoning": {"type": "string"},
            "hypothesis": {"type": "string"},
        },
        "required": [
            "dataset",
            "comparison_mode",
            "steps",
            "time_budget_s",
            "seed",
            "models",
            "reasoning",
            "hypothesis",
        ],
        "additionalProperties": False,
    }


def _build_strategist_prompt(
    candidate_payloads: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    state: dict[str, object],
) -> str:
    compact_leaderboard = [
        {
            "model": row["model"],
            "mean_relative_score": round(float(row["mean_relative_score"]), 4),
            "coverage_runs": int(row["coverage_runs"]),
        }
        for row in leaderboard_rows[:6]
    ]
    recent_iterations = _load_recent_loop_iterations(state)
    return (
        "You are the ABPT autoresearch strategist.\n"
        "Goal: choose the next architecture experiment that maximizes information gain about which small-model architecture generalizes best across TinyStories and code.\n\n"
        "Rules:\n"
        "- Choose EXACTLY ONE candidate_id from the allowed catalog.\n"
        "- Prefer underexplored seeds, cross-domain balance, and experiments that discriminate between strong contenders.\n"
        "- Treat this as architecture research, not code-only tuning.\n"
        "- Output only JSON matching the schema.\n\n"
        f"Leaderboard summary:\n{json.dumps(compact_leaderboard, ensure_ascii=False)}\n\n"
        f"Recent loop history:\n{json.dumps(recent_iterations, ensure_ascii=False)}\n\n"
        f"Allowed candidates:\n{json.dumps(candidate_payloads, ensure_ascii=False)}\n"
    )


def _summarize_result_coverage(results: list[dict[str, object]], limit: int = 16) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for result in results[-limit:]:
        runtime = result.get("runtime")
        dataset = result.get("dataset")
        if not isinstance(runtime, dict) or not isinstance(dataset, dict):
            continue
        summary.append(
            {
                "dataset": "code" if str(dataset.get("name")) == "the-stack-bpe" else "tinystories",
                "mode": "time" if float(runtime.get("time_budget_s", 0.0)) > 0.0 else "steps",
                "seed": int(runtime.get("seed", 42)),
                "models": list(runtime.get("models", [])),
            }
        )
    return summary


def _build_open_strategist_prompt(
    *,
    state: dict[str, object],
    results: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    allowed_models: tuple[str, ...],
    allowed_seeds: tuple[int, ...],
) -> str:
    compact_leaderboard = [
        {
            "model": row["model"],
            "mean_relative_score": round(float(row["mean_relative_score"]), 4),
            "coverage_runs": int(row["coverage_runs"]),
        }
        for row in leaderboard_rows[:8]
    ]
    coverage = _summarize_result_coverage(results)
    recent_iterations = _load_recent_loop_iterations(state)
    return (
        "You are the ABPT open autoresearch strategist.\n"
        "Your job is to choose the next experiment spec, not just pick from a shortlist.\n\n"
        "Research goal:\n"
        "- discover which architecture generalizes best across TinyStories and code\n"
        "- prefer architecture-level signal over domain-specific hacks\n"
        "- but exploit code-specific clues when they help discriminate candidate mechanisms\n\n"
        "Freedom rules:\n"
        "- You may choose dataset, mode (steps/time), seed, and a focused subset of models.\n"
        "- You may narrow the model set to the most informative contenders.\n"
        "- Always include `uniform` as the baseline.\n"
        "- Choose 1-5 non-baseline models.\n"
        "- Keep the benchmark fair: batch_size=4, seq_len=64, TinyStories rows fixed, code bytes fixed.\n"
        "- Use steps only from [60, 100, 150, 200].\n"
        "- Use time budgets only from [20, 30, 40].\n"
        "- Prefer seeds from the allowed list that are underexplored.\n"
        "- Avoid repeating an already-run dataset/mode/seed/model-set if another informative choice exists.\n"
        "- Output only JSON matching the schema.\n\n"
        f"Allowed models: {json.dumps(list(allowed_models), ensure_ascii=False)}\n"
        f"Allowed seeds: {json.dumps(list(allowed_seeds), ensure_ascii=False)}\n\n"
        f"Leaderboard summary:\n{json.dumps(compact_leaderboard, ensure_ascii=False)}\n\n"
        f"Recent executed loop iterations:\n{json.dumps(recent_iterations, ensure_ascii=False)}\n\n"
        f"Recent benchmark coverage:\n{json.dumps(coverage, ensure_ascii=False)}\n"
    )


def _parse_json_output(raw: str) -> dict[str, object] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _call_codex_strategist(
    prompt: str,
    schema: dict[str, object] | None,
    *,
    timeout: int,
    model: str,
) -> tuple[dict[str, object] | None, str | None]:
    codex_bin = shutil.which("codex")
    if not codex_bin:
        return None, None
    output_file = Path(tempfile.mkstemp(prefix="abpt_codex_last_", suffix=".txt")[1])
    schema_file: Path | None = None
    if schema is not None:
        schema_file = Path(tempfile.mkstemp(prefix="abpt_codex_schema_", suffix=".json")[1])
        schema_file.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--output-last-message",
        str(output_file),
        "-m",
        model,
        "-c",
        "model_reasoning_effort='low'",
        "-",
    ]
    if schema_file is not None:
        cmd[5:5] = ["--output-schema", str(schema_file)]
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=ROOT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, None
    raw = None
    if output_file.exists():
        raw = output_file.read_text(encoding="utf-8", errors="replace")
    elif result.stdout:
        raw = result.stdout
    parsed = _parse_json_output(raw or "")
    try:
        output_file.unlink(missing_ok=True)
        if schema_file is not None:
            schema_file.unlink(missing_ok=True)
    except OSError:
        pass
    return parsed, raw


def _call_deepseek_strategist(
    prompt: str,
    *,
    timeout: int,
    model: str,
) -> tuple[dict[str, object] | None, str | None]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None, None
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        url=os.environ.get("STRATEGIST_DEEPSEEK_URL", "https://api.deepseek.com/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_response = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return None, None
    try:
        parsed_outer = json.loads(raw_response)
        content = parsed_outer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None, raw_response
    parsed = _parse_json_output(content)
    return parsed, content


def _select_backend(preferred: str) -> list[str]:
    preferred = preferred.lower().strip()
    if preferred == "rule":
        return ["rule"]
    if preferred == "codex":
        return ["codex", "rule"]
    if preferred == "deepseek":
        return ["deepseek", "codex", "rule"]
    if preferred == "auto":
        order: list[str] = []
        if os.environ.get("DEEPSEEK_API_KEY"):
            order.append("deepseek")
        order.append("codex")
        order.append("rule")
        return order
    return ["codex", "rule"]


def _normalize_open_models(raw_models: list[object] | tuple[object, ...], allowed_models: tuple[str, ...]) -> tuple[str, ...] | None:
    seen: list[str] = []
    for item in raw_models:
        name = str(item).strip()
        if not name or name not in allowed_models or name in seen:
            continue
        seen.append(name)
    if "uniform" not in seen:
        seen.insert(0, "uniform")
    if len(seen) < 2 or len(seen) > 6:
        return None
    return tuple(seen)


def _coerce_open_choice_to_spec(
    parsed: dict[str, object],
    *,
    allowed_models: tuple[str, ...],
    allowed_seeds: tuple[int, ...],
    batch_size: int,
    seq_len: int,
    eval_every: int,
    eval_steps: int,
    train_rows: int,
    val_rows: int,
    code_repo: str,
    code_lang: str,
    code_bytes: int,
) -> ExperimentSpec | None:
    dataset = str(parsed.get("dataset", "")).strip()
    mode = str(parsed.get("comparison_mode", parsed.get("mode", ""))).strip()
    if dataset not in {"tinystories", "code"} or mode not in {"steps", "time"}:
        return None
    try:
        seed = int(parsed.get("seed"))
    except (TypeError, ValueError):
        return None
    if seed not in allowed_seeds:
        return None

    raw_models = parsed.get("models")
    if not isinstance(raw_models, list):
        return None
    models = _normalize_open_models(raw_models, allowed_models)
    if models is None:
        return None

    if mode == "steps":
        try:
            steps = int(parsed.get("steps"))
        except (TypeError, ValueError):
            return None
        if steps not in OPEN_ALLOWED_STEP_BUDGETS:
            return None
        time_budget_s = 0.0
    else:
        try:
            time_budget_s = float(parsed.get("time_budget_s", parsed.get("time_budget")))
        except (TypeError, ValueError):
            return None
        if time_budget_s not in OPEN_ALLOWED_TIME_BUDGETS:
            return None
        steps = 9999

    return ExperimentSpec(
        name=f"{dataset}_open_{mode}",
        dataset=dataset,
        models=models,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        eval_every=eval_every,
        eval_steps=eval_steps,
        train_rows=train_rows,
        val_rows=val_rows,
        time_budget_s=time_budget_s,
        code_repo=code_repo,
        code_lang=code_lang,
        code_bytes=code_bytes,
        seed=seed,
    )


def _strategist_select(
    ranked_candidates: list[object],
    results: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    state: dict[str, object],
    *,
    backend_preference: str,
    timeout: int,
    model: str,
    top_n: int,
    strategist_style: str,
    allowed_models: tuple[str, ...],
    allowed_seeds: tuple[int, ...],
    batch_size: int,
    seq_len: int,
    eval_every: int,
    eval_steps: int,
    train_rows: int,
    val_rows: int,
    code_repo: str,
    code_lang: str,
    code_bytes: int,
) -> tuple[object, str, dict[str, object] | None]:
    if not ranked_candidates:
        raise ValueError("No candidates to select from")
    if strategist_style == "open":
        open_prompt = _build_open_strategist_prompt(
            state=state,
            results=results,
            leaderboard_rows=leaderboard_rows,
            allowed_models=allowed_models,
            allowed_seeds=allowed_seeds,
        )
        for backend in _select_backend(backend_preference):
            if backend == "rule":
                break
            if backend == "codex":
                parsed, raw = _call_codex_strategist(open_prompt, None, timeout=timeout, model=model)
            else:
                deepseek_model = os.environ.get("STRATEGIST_DEEPSEEK_MODEL", "deepseek-chat")
                parsed, raw = _call_deepseek_strategist(open_prompt, timeout=timeout, model=deepseek_model)
            if isinstance(parsed, dict):
                coerced = _coerce_open_choice_to_spec(
                    parsed,
                    allowed_models=allowed_models,
                    allowed_seeds=allowed_seeds,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    eval_every=eval_every,
                    eval_steps=eval_steps,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    code_repo=code_repo,
                    code_lang=code_lang,
                    code_bytes=code_bytes,
                )
                if coerced is not None:
                    for item in ranked_candidates:
                        if isinstance(item, type(ranked_candidates[0])) and item.spec == coerced:
                            parsed["raw_backend_output"] = raw
                            return item, backend + "_open", parsed
                    from src.utils.autoresearch import SelectionDecision

                    synthetic = SelectionDecision(
                        spec=coerced,
                        pending_models=tuple(model_name for model_name in coerced.models if model_name != "uniform"),
                        score=0.0,
                        reason="open_strategist_selected_custom_spec",
                    )
                    parsed["raw_backend_output"] = raw
                    return synthetic, backend + "_open", parsed
    shortlist = ranked_candidates[: max(1, top_n)]
    candidate_payloads = [_candidate_payload(item) for item in shortlist]
    payload_map = {item["candidate_id"]: shortlist[idx] for idx, item in enumerate(candidate_payloads)}
    prompt = _build_strategist_prompt(candidate_payloads, leaderboard_rows, state)
    schema = _candidate_schema(list(payload_map.keys()))

    for backend in _select_backend(backend_preference):
        if backend == "rule":
            return shortlist[0], "rule", None
        if backend == "codex":
            parsed, raw = _call_codex_strategist(prompt, schema, timeout=timeout, model=model)
        else:
            deepseek_model = os.environ.get("STRATEGIST_DEEPSEEK_MODEL", "deepseek-chat")
            parsed, raw = _call_deepseek_strategist(prompt, timeout=timeout, model=deepseek_model)
        if isinstance(parsed, dict):
            candidate_id = str(parsed.get("candidate_id", "")).strip()
            if candidate_id in payload_map:
                parsed["raw_backend_output"] = raw
                return payload_map[candidate_id], backend, parsed
    return shortlist[0], "rule_fallback", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture-aware FOG autoresearch loop")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/autoresearch")
    parser.add_argument("--runner", type=str, default="scripts/run_tinystories_hypothesis_compare.py")
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--tinystories_steps", type=int, default=150)
    parser.add_argument("--code_steps", type=int, default=100)
    parser.add_argument("--time_budget_s", type=float, default=20.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--train_rows", type=int, default=1000)
    parser.add_argument("--val_rows", type=int, default=100)
    parser.add_argument("--code_repo", type=str, default="bigcode/the-stack-smol-xs")
    parser.add_argument("--code_lang", type=str, default="python")
    parser.add_argument("--code_bytes", type=int, default=1_200_000)
    parser.add_argument("--seed_schedule", type=str, default=_default_seed_list())
    parser.add_argument("--strategist", type=str, default="rule", choices=["rule", "codex", "deepseek", "auto"])
    parser.add_argument("--strategist_style", type=str, default="open", choices=["bounded", "open"])
    parser.add_argument("--strategist_model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--strategist_timeout", type=int, default=120)
    parser.add_argument("--strategist_top_n", type=int, default=12)
    parser.add_argument(
        "--frontier_models",
        nargs="+",
        default=list(DEFAULT_FRONTIER_MODELS),
    )
    args = parser.parse_args()

    results_dir = (ROOT / args.results_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    runner_path = (ROOT / args.runner).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(output_dir)
    deadline = time.time() + args.hours * 3600.0 if args.hours > 0 else None
    remaining_iterations = args.iterations

    while True:
        if deadline is not None and time.time() >= deadline:
            print("[autoresearch] time budget reached, stopping loop")
            break
        if deadline is None and remaining_iterations <= 0:
            break
        results = load_benchmark_results(results_dir)
        specs = build_frontier_specs(
            frontier_models=tuple(args.frontier_models),
            tinystories_steps=args.tinystories_steps,
            code_steps=args.code_steps,
            time_budget_s=args.time_budget_s,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_every=args.eval_every,
            eval_steps=args.eval_steps,
            train_rows=args.train_rows,
            val_rows=args.val_rows,
            code_repo=args.code_repo,
            code_lang=args.code_lang,
            code_bytes=args.code_bytes,
            seeds=_parse_seed_list(args.seed_schedule),
        )
        leaderboard_rows = build_global_leaderboard(results, frontier_models=tuple(args.frontier_models))
        _save_leaderboard(output_dir, leaderboard_rows)
        ranked_candidates = rank_candidate_experiments(specs=specs, results=results)

        if not ranked_candidates:
            print("No pending frontier experiment found.")
            break
        decision, strategist_backend, strategist_payload = _strategist_select(
            ranked_candidates=ranked_candidates,
            results=results,
            leaderboard_rows=leaderboard_rows,
            state=state,
            backend_preference=args.strategist,
            timeout=args.strategist_timeout,
            model=args.strategist_model,
            top_n=args.strategist_top_n,
            strategist_style=args.strategist_style,
            allowed_models=OPEN_MODEL_CATALOG,
            allowed_seeds=_parse_seed_list(args.seed_schedule),
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_every=args.eval_every,
            eval_steps=args.eval_steps,
            train_rows=args.train_rows,
            val_rows=args.val_rows,
            code_repo=args.code_repo,
            code_lang=args.code_lang,
            code_bytes=args.code_bytes,
        )

        run_iteration = len(state["iterations"]) + 1
        output_json = output_dir / f"{run_iteration:02d}_{decision.spec.slug()}.json"
        command = build_command(
            spec=decision.spec,
            runner_path=runner_path,
            output_path=output_json,
            python_executable=args.python_executable,
        )

        record = {
            "iteration": run_iteration,
            "selected_spec": asdict(decision.spec),
            "pending_models": list(decision.pending_models),
            "selection_score": decision.score,
            "selection_reason": decision.reason,
            "strategist_backend": strategist_backend,
            "strategist_payload": strategist_payload,
            "output_json": str(output_json),
            "command": command,
            "created_at": _timestamp(),
            "status": "planned" if args.dry_run else "running",
        }
        state["iterations"].append(record)
        _save_state(output_dir, state)

        print(f"[autoresearch] iteration={run_iteration}")
        print(f"[autoresearch] selected={decision.spec.name}")
        print(f"[autoresearch] pending={', '.join(decision.pending_models)}")
        print(f"[autoresearch] reason={decision.reason}")
        print(f"[autoresearch] strategist={strategist_backend}")
        if isinstance(strategist_payload, dict):
            print(f"[autoresearch] strategist_hypothesis={strategist_payload.get('hypothesis', '')}")
        print(f"[autoresearch] output={output_json}")
        print(f"[autoresearch] command={' '.join(command)}")

        if args.dry_run:
            record["status"] = "dry_run"
            record["finished_at"] = _timestamp()
            _save_state(output_dir, state)
            break

        subprocess.run(command, cwd=ROOT, check=True)

        fresh_results = load_benchmark_results(results_dir)
        leaderboard_rows = build_global_leaderboard(fresh_results, frontier_models=tuple(args.frontier_models))
        _save_leaderboard(output_dir, leaderboard_rows)
        record["status"] = "completed"
        record["finished_at"] = _timestamp()
        record["leaderboard_after_run"] = leaderboard_rows
        _save_state(output_dir, state)
        if deadline is None:
            remaining_iterations -= 1


if __name__ == "__main__":
    main()
