"""
Remote Worker — вызывает эксперименты на HF Space (Gradio + ZeroGPU) через API.

Использует gradio_client для надёжного вызова Gradio API любой версии.

Переменные окружения:
    WORKER_URL — URL HF Space (например: https://kharki-abpt.hf.space)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


HF_SPACE_REPO = Path(__file__).resolve().parents[1].parent / "hf_abpt_space"
DEFAULT_WORKER_URL = "https://kharki-abpt.hf.space"
DEFAULT_SPACE_ID = "kharki/abpt"


def _infer_worker_url() -> str:
    worker_url = os.environ.get("WORKER_URL", "").rstrip("/")
    if worker_url:
        return worker_url

    if HF_SPACE_REPO.exists():
        import subprocess as _sp

        try:
            result = _sp.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(HF_SPACE_REPO),
                capture_output=True,
                text=True,
                timeout=10,
            )
            remote = (result.stdout or "").strip()
            if remote.startswith("https://huggingface.co/spaces/"):
                slug = remote.removeprefix("https://huggingface.co/spaces/").strip("/")
                if slug:
                    return f"https://{slug.replace('/', '-')}.hf.space"
        except Exception:
            pass

    return DEFAULT_WORKER_URL


def _infer_space_id() -> str | None:
    if HF_SPACE_REPO.exists():
        import subprocess as _sp

        try:
            result = _sp.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(HF_SPACE_REPO),
                capture_output=True,
                text=True,
                timeout=10,
            )
            remote = (result.stdout or "").strip()
            prefix = "https://huggingface.co/spaces/"
            if remote.startswith(prefix):
                slug = remote.removeprefix(prefix).strip("/")
                return slug or DEFAULT_SPACE_ID
        except Exception:
            pass
    return DEFAULT_SPACE_ID


def _fetch_space_runtime() -> dict[str, Any] | None:
    try:
        import requests
    except ImportError:
        return None

    space_id = _infer_space_id()
    if not space_id:
        return None
    try:
        response = requests.get(
            f"https://huggingface.co/api/spaces/{space_id}/runtime",
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def remote_worker_run(
    script: str,
    args: dict[str, Any],
    model: str = "Qwen/Qwen3.5-4B",
    timeout: int = 360,
) -> dict[str, Any]:
    """
    Вызывает Gradio API на HF Space через gradio_client.
    Автоматически синхронизирует новые скрипты на Space.
    """
    worker_url = _infer_worker_url()
    if not worker_url:
        return {"status": "error", "error": "WORKER_URL not set"}

    runtime = _fetch_space_runtime()
    if runtime:
        hardware = runtime.get("hardware", {})
        current_hardware = str(hardware.get("current", "unknown"))
        requested_hardware = str(hardware.get("requested", "unknown"))
        stage = str(runtime.get("stage", "unknown"))
        print(
            f"[RemoteWorker] Space runtime: stage={stage}, "
            f"hardware.current={current_hardware}, hardware.requested={requested_hardware}"
        )
        allow_cpu_space = os.environ.get("ALLOW_CPU_SPACE", "").strip() == "1"
        if not allow_cpu_space and "gpu" not in current_hardware.lower():
            return {
                "status": "error",
                "error": (
                    "HF Space is not on GPU hardware "
                    f"(current={current_hardware}, requested={requested_hardware}, stage={stage}). "
                    "Switch the Space to ZeroGPU/GPU or set ALLOW_CPU_SPACE=1 to force a CPU run."
                ),
            }

    # Sync new script to Space if needed
    _sync_script_to_space(script)

    try:
        from gradio_client import Client
    except ImportError:
        return {"status": "error", "error": "pip install gradio_client"}

    request_json = json.dumps({
        "script": script,
        "args": args,
        "model": model,
        "timeout": min(timeout, 300),
    })

    print(f"[RemoteWorker] Connecting to {worker_url} ...")

    try:
        client = Client(worker_url)
    except Exception as e:
        return {"status": "error", "error": f"Cannot connect to Space: {e}"}

    print(f"[RemoteWorker] Running {script} on remote GPU...")
    t0 = time.time()

    try:
        result_str = client.predict(
            request_json,
            api_name="/run",
        )
    except Exception as e:
        return {"status": "error", "error": f"API call failed: {e}"}

    elapsed = time.time() - t0
    print(f"[RemoteWorker] Response in {elapsed:.1f}s")

    try:
        result = json.loads(result_str)
    except (json.JSONDecodeError, TypeError) as e:
        return {"status": "error", "error": f"Bad response: {e}", "raw": str(result_str)[:500]}

    print(f"[RemoteWorker] status={result.get('status')}, elapsed={result.get('elapsed_seconds')}s")

    # Save result_json locally
    if result.get("result_json"):
        _save_result_locally(script, result["result_json"])

    result.setdefault("output_file", None)
    return result


def _sync_script_to_space(script: str) -> None:
    """Синхронизирует целевой скрипт и shared src/ код в HF Space repo."""
    import subprocess as _sp

    local_script = Path(__file__).resolve().parent / script
    space_script = HF_SPACE_REPO / "scripts" / script
    local_src = Path(__file__).resolve().parents[1] / "src"
    space_src = HF_SPACE_REPO / "src"
    local_app = Path(__file__).resolve().parents[1] / "hf_space" / "app.py"
    space_app = HF_SPACE_REPO / "app.py"
    local_readme = Path(__file__).resolve().parents[1] / "hf_space" / "README.md"
    space_readme = HF_SPACE_REPO / "README.md"
    local_requirements = Path(__file__).resolve().parents[1] / "hf_space" / "requirements.txt"
    space_requirements = HF_SPACE_REPO / "requirements.txt"

    if not local_script.exists():
        return
    if not HF_SPACE_REPO.exists():
        print(f"[RemoteWorker] Space repo not found at {HF_SPACE_REPO}, skip sync")
        return
    import shutil
    changed = False
    if not space_script.exists() or space_script.read_bytes() != local_script.read_bytes():
        shutil.copy2(str(local_script), str(space_script))
        changed = True
    if local_app.exists() and (not space_app.exists() or space_app.read_bytes() != local_app.read_bytes()):
        shutil.copy2(str(local_app), str(space_app))
        changed = True
    if local_readme.exists() and (not space_readme.exists() or space_readme.read_bytes() != local_readme.read_bytes()):
        shutil.copy2(str(local_readme), str(space_readme))
        changed = True
    if local_requirements.exists() and (
        not space_requirements.exists() or space_requirements.read_bytes() != local_requirements.read_bytes()
    ):
        shutil.copy2(str(local_requirements), str(space_requirements))
        changed = True
    if local_src.exists():
        shutil.copytree(local_src, space_src, dirs_exist_ok=True)
        changed = True
    if not changed:
        return
    print(f"[RemoteWorker] Syncing {script} (+ worker app + shared src) to HF Space...")

    try:
        add_targets = [f"scripts/{script}", "src"]
        if local_app.exists():
            add_targets.append("app.py")
        if local_readme.exists():
            add_targets.append("README.md")
        if local_requirements.exists():
            add_targets.append("requirements.txt")
        _sp.run(["git", "add", *add_targets], cwd=str(HF_SPACE_REPO),
                capture_output=True, timeout=10)
        status = _sp.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(HF_SPACE_REPO),
            capture_output=True,
            timeout=10,
        )
        if status.returncode == 0:
            print("[RemoteWorker] No staged changes after sync")
            return
        _sp.run(["git", "commit", "-m", f"auto: sync {script}"], cwd=str(HF_SPACE_REPO),
                capture_output=True, timeout=10)
        result = _sp.run(["git", "push"], cwd=str(HF_SPACE_REPO),
                         capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"[RemoteWorker] Pushed {script} to Space. Waiting 15s for rebuild...")
            time.sleep(15)  # wait for Space to pick up new code
        else:
            print(f"[RemoteWorker] Push failed: {result.stderr[-200:]}")
    except Exception as e:
        print(f"[RemoteWorker] Sync error: {e}")


def _save_result_locally(script: str, result_json: dict) -> None:
    archive_dir = Path(__file__).resolve().parents[1] / "archive"
    archive_dir.mkdir(exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    name = script.replace(".py", "")
    out_path = archive_dir / f"{name}_{ts}_remote.json"
    out_path.write_text(json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RemoteWorker] Result saved: {out_path}")
