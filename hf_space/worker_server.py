"""
ABPT Remote Worker — FastAPI сервер на HF Space с GPU.
Принимает запросы от локального оркестратора, запускает эксперименты.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ABPT Research Worker")

SCRIPTS_DIR = Path("/app/scripts")
RESULTS_DIR = Path("/data/results") if Path("/data").exists() else Path("/tmp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")


class RunRequest(BaseModel):
    script: str
    args: dict[str, Any] = {}
    model: str = "Qwen/Qwen3.5-4B"
    timeout: int = 3600


class RunResponse(BaseModel):
    status: str
    returncode: int = -1
    elapsed_seconds: float = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    result_json: dict[str, Any] | None = None


def _verify_token(authorization: str | None) -> None:
    if not WORKER_TOKEN:
        return  # no token configured = open access
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    if authorization[7:] != WORKER_TOKEN:
        raise HTTPException(403, "Invalid token")


@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scripts": [p.name for p in SCRIPTS_DIR.glob("run_qwen_*.py")],
    }


@app.post("/run", response_model=RunResponse)
def run_experiment(
    req: RunRequest,
    authorization: str | None = Header(None),
):
    _verify_token(authorization)

    script_path = SCRIPTS_DIR / req.script
    if not script_path.exists():
        raise HTTPException(404, f"Script not found: {req.script}")

    # Sanitize: script must be in SCRIPTS_DIR
    if not str(script_path.resolve()).startswith(str(SCRIPTS_DIR.resolve())):
        raise HTTPException(400, "Invalid script path")

    # Build command
    cmd = [sys.executable, str(script_path)]
    for key, val in req.args.items():
        cli_key = f"--{key.replace('_', '-')}"
        if isinstance(val, (list, tuple)):
            cmd.append(cli_key)
            cmd.extend(str(item) for item in val)
        else:
            cmd += [cli_key, str(val)]
    cmd += ["--model-name", req.model]

    print(f"[Worker] Running: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR.parent),
            timeout=req.timeout,
            env={**os.environ, "TRANSFORMERS_CACHE": str(Path(os.environ.get("HF_HOME", "/tmp")) / "hub")},
        )
    except subprocess.TimeoutExpired:
        return RunResponse(status="timeout", elapsed_seconds=req.timeout)
    except Exception as e:
        return RunResponse(status="error", stderr_tail=str(e))

    elapsed = time.time() - t0

    # Try to find and read result JSON
    result_json = _extract_result_json(result.stdout or "", result.stderr or "")

    return RunResponse(
        status="success" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        elapsed_seconds=round(elapsed, 1),
        stdout_tail=(result.stdout or "")[-3000:],
        stderr_tail=(result.stderr or "")[-1000:],
        result_json=result_json,
    )


def _extract_result_json(stdout: str, stderr: str) -> dict[str, Any] | None:
    """Find saved JSON path in output, read and return it."""
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
    return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    print(f"[Worker] Starting on port {port}")
    print(f"[Worker] Scripts dir: {SCRIPTS_DIR}")
    print(f"[Worker] Results dir: {RESULTS_DIR}")
    print(f"[Worker] Token configured: {'yes' if WORKER_TOKEN else 'no (open access)'}")
    uvicorn.run(app, host="0.0.0.0", port=port)
