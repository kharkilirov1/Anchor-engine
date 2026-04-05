"""
ABPT Research Worker — Gradio app с ZeroGPU.
Бесплатный T4 GPU на HuggingFace Spaces.
"""
from __future__ import annotations

import io
import json
import os
import re
import runpy
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import gradio as gr
import spaces

SCRIPTS_DIR = Path(__file__).parent / "scripts"
INPROCESS_CACHEABLE_SCRIPTS = {
    "run_qwen_per_case_diagnostic.py",
    "run_qwen_per_case_diagnostic_v2.py",
}
_OVERLAY_CACHE: dict[str, Any] = {}
SCRIPT_CLI_METADATA: dict[str, dict[str, Any]] = {
    "run_qwen_anchor_carryover_probe.py": {
        "model_flag": "--model",
        "preserve_underscores": {"max_length", "neutral_components", "neutral_variance_cutoff", "case_name"},
    },
    "run_qwen_anchor_layer_profile_map.py": {
        "model_flag": "--model",
        "preserve_underscores": {"max_length", "case_name"},
    },
    "run_qwen_anchor_concept_direction_map.py": {
        "model_flag": "--model",
        "preserve_underscores": {"max_length", "neutral_components", "neutral_variance_cutoff", "case_name"},
    },
    "run_qwen_future_influence_probe.py": {
        "model_flag": "--model",
        "preserve_underscores": {"max_length", "future_window", "top_k", "span_threshold", "top_spans", "case_filter"},
    },
}


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _default_cpu_quant_mode() -> str:
    return os.environ.get("CPU_QUANT_MODE", "int8").strip().lower() or "int8"


def _build_cpu_quant_kwargs(device: str | None) -> tuple[dict[str, Any], str | None]:
    if str(device).lower() != "cpu":
        return {}, None
    quant_mode = _default_cpu_quant_mode()
    if quant_mode in {"off", "none", "false", "0"}:
        return {}, None
    try:
        from transformers import TorchAoConfig
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
    except Exception as exc:
        return {}, f"cpu_quant_unavailable: {exc}"
    try:
        if quant_mode == "int4":
            quantization_config = TorchAoConfig(Int4WeightOnlyConfig(group_size=128))
        else:
            quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
        return {
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
        }, quant_mode
    except Exception as exc:
        return {}, f"cpu_quant_init_failed: {exc}"


def _build_cli_args(script: str, args: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    meta = SCRIPT_CLI_METADATA.get(script, {})
    preserve_underscores = set(meta.get("preserve_underscores", set()))
    for key, val in args.items():
        key_str = str(key)
        cli_key = f"--{key_str}" if key_str in preserve_underscores else f"--{key_str.replace('_', '-')}"
        if isinstance(val, (list, tuple)):
            cli_args.append(cli_key)
            cli_args.extend(str(item) for item in val)
        else:
            cli_args.extend([cli_key, str(val)])
    return cli_args


def _load_cached_overlay(
    model_name: str,
    *,
    factory: Any,
    cfg: Any,
    device: str | None,
    torch_dtype: Any,
    extra_kwargs: dict[str, Any],
) -> Any:
    import torch
    from src.model.qwen_anchor_overlay import QwenAnchorOverlay

    resolved_device = device or ("cuda" if torch.cuda.is_available() else None)
    resolved_dtype = torch_dtype
    if resolved_dtype is None and torch.cuda.is_available():
        resolved_dtype = torch.float16
    cpu_quant_kwargs, cpu_quant_status = _build_cpu_quant_kwargs(resolved_device)
    cache_key = f"{model_name}|{resolved_device}|{resolved_dtype}|{cpu_quant_status or 'fp'}"
    cached = _OVERLAY_CACHE.get(cache_key)
    if cached is not None:
        if resolved_device is not None:
            cached = cached.to(resolved_device)
        cached.eval()
        return cached
    load_kwargs = dict(extra_kwargs)
    load_kwargs.update(cpu_quant_kwargs)

    if hasattr(factory, "__func__"):
        overlay = factory.__func__(
            QwenAnchorOverlay,
            model_name=model_name,
            cfg=cfg,
            device=resolved_device,
            torch_dtype=resolved_dtype,
            **load_kwargs,
        )
    else:
        overlay = factory(
            model_name=model_name,
            cfg=cfg,
            device=resolved_device,
            torch_dtype=resolved_dtype,
            **load_kwargs,
        )
    overlay.eval()
    setattr(overlay, "_cpu_quant_status", cpu_quant_status or "fp")
    _OVERLAY_CACHE[cache_key] = overlay
    return overlay


def _run_inprocess_cached(
    *,
    script_path: Path,
    script: str,
    args: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    from src.model.qwen_anchor_overlay import QwenAnchorOverlay

    original_factory = QwenAnchorOverlay.from_pretrained
    original_argv = list(sys.argv)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    def cached_from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3.5-4B",
        cfg: Any | None = None,
        device: str | None = None,
        torch_dtype: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        return _load_cached_overlay(
            model_name=model_name,
            factory=original_factory,
            cfg=cfg,
            device=device,
            torch_dtype=torch_dtype,
            extra_kwargs=kwargs,
        )

    model_flag = str(SCRIPT_CLI_METADATA.get(script, {}).get("model_flag", "--model-name"))
    cli_args = [str(script_path), *_build_cli_args(script, args), model_flag, model]
    returncode = 0
    t0 = time.time()
    try:
        QwenAnchorOverlay.from_pretrained = classmethod(cached_from_pretrained)
        sys.argv = cli_args
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                runpy.run_path(str(script_path), run_name="__main__")
            except SystemExit as exc:
                code = exc.code
                if code in (None, 0):
                    returncode = 0
                elif isinstance(code, int):
                    returncode = code
                else:
                    returncode = 1
                    print(code, file=sys.stderr)
            except Exception:
                returncode = 1
                traceback.print_exc()
    finally:
        QwenAnchorOverlay.from_pretrained = original_factory
        sys.argv = original_argv

    stdout = stdout_buffer.getvalue()
    stderr = stderr_buffer.getvalue()
    return {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "elapsed_seconds": round(time.time() - t0, 1),
        "stdout_tail": stdout[-3000:],
        "stderr_tail": stderr[-1000:],
        "result_json": _extract_result_json(stdout, stderr),
        "runner": "inprocess_cached",
        "cached_models": len(_OVERLAY_CACHE),
    }


@spaces.GPU(duration=300)  # до 5 минут GPU на один вызов
def run_experiment(request_json: str) -> str:
    """
    Запускает эксперимент на GPU. Принимает и возвращает JSON строки.

    Input JSON:
        {"script": "run_qwen_phase_probe.py", "args": {...}, "model": "Qwen/Qwen3.5-4B"}

    Output JSON:
        {"status": "success", "returncode": 0, "elapsed_seconds": 120,
         "stdout_tail": "...", "stderr_tail": "...", "result_json": {...}}
    """
    try:
        req = json.loads(request_json)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "stderr_tail": f"Invalid JSON: {e}"})

    script = req.get("script", "")
    args = dict(req.get("args", {}))
    model = req.get("model", "Qwen/Qwen3.5-4B")
    timeout = min(req.get("timeout", 300), 300)  # ZeroGPU max ~300s
    args.setdefault("device", _default_device())

    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        return json.dumps({"status": "error", "stderr_tail": f"Script not found: {script}"})

    # Sanitize
    if ".." in script or "/" in script:
        return json.dumps({"status": "error", "stderr_tail": "Invalid script name"})

    if script in INPROCESS_CACHEABLE_SCRIPTS:
        print(f"[Worker] Running cached in-process path for {script}")
        try:
            output = _run_inprocess_cached(
                script_path=script_path,
                script=script,
                args=args,
                model=model,
            )
            return json.dumps(output, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "stderr_tail": f"in-process runner failed: {e}"})

    # Build command
    model_flag = str(SCRIPT_CLI_METADATA.get(script, {}).get("model_flag", "--model-name"))
    cmd = [sys.executable, str(script_path), *_build_cli_args(script, args), model_flag, model]

    print(f"[Worker] Running: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR.parent),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "timeout", "elapsed_seconds": timeout})
    except Exception as e:
        return json.dumps({"status": "error", "stderr_tail": str(e)})

    elapsed = time.time() - t0

    # Try to find and read result JSON
    result_json = _extract_result_json(result.stdout or "", result.stderr or "")

    output = {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "stdout_tail": (result.stdout or "")[-3000:],
        "stderr_tail": (result.stderr or "")[-1000:],
        "result_json": result_json,
    }

    return json.dumps(output, ensure_ascii=False)


def health_check() -> str:
    """Проверка состояния worker."""
    import torch
    scripts = [p.name for p in SCRIPTS_DIR.glob("run_qwen_*.py")] if SCRIPTS_DIR.exists() else []
    return json.dumps({
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "default_device": _default_device(),
        "cpu_quant_mode": _default_cpu_quant_mode(),
        "scripts": scripts,
        "cached_models": len(_OVERLAY_CACHE),
        "inprocess_cacheable_scripts": sorted(INPROCESS_CACHEABLE_SCRIPTS),
    }, indent=2)


def _extract_result_json(stdout: str, stderr: str) -> dict[str, Any] | None:
    combined = stdout + "\n" + stderr
    for pattern in (
        r"saved_json=(?P<path>[^\r\n]+)",
        r"saved json:\s*(?P<path>[^\r\n]+)",
        r"saved_json:\s*(?P<path>[^\r\n]+)",
    ):
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            raw_path = match.group("path").strip().strip("'\"")
            p = Path(raw_path)
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
    marker = "===FINAL_RESULT==="
    marker_index = combined.rfind(marker)
    if marker_index >= 0:
        tail = combined[marker_index + len(marker):].strip()
        decoder = json.JSONDecoder()
        for i, ch in enumerate(tail):
            if ch not in "{[":
                continue
            try:
                payload, _ = decoder.raw_decode(tail[i:])
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return None


# Gradio UI + API
with gr.Blocks(title="ABPT Research Worker") as demo:
    gr.Markdown("# ABPT Research Worker (ZeroGPU)")
    gr.Markdown("GPU worker for autonomous research loop. Use via API.")

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="Request JSON",
                placeholder='{"script": "run_qwen_phase_probe.py", "args": {}, "model": "Qwen/Qwen3.5-4B"}',
                lines=5,
            )
            run_btn = gr.Button("Run Experiment", variant="primary")
        with gr.Column():
            output_box = gr.Textbox(label="Result JSON", lines=10)

    run_btn.click(fn=run_experiment, inputs=input_box, outputs=output_box, api_name="run")

    health_btn = gr.Button("Health Check")
    health_output = gr.Textbox(label="Health", lines=5)
    health_btn.click(fn=health_check, outputs=health_output, api_name="health")

demo.launch()
