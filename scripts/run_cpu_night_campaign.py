from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARCHIVE_DIR = ROOT / "archive"
ARCHIVE_DIR.mkdir(exist_ok=True)


def _extract_summary(result: dict[str, Any]) -> dict[str, Any]:
    stdout_tail = str(result.get("stdout_tail", ""))
    final_marker = "===FINAL_RESULT==="
    payload: dict[str, Any] | None = None
    if final_marker in stdout_tail:
        fragment = stdout_tail.split(final_marker, 1)[-1].strip()
        try:
            payload = json.loads(fragment)
        except json.JSONDecodeError:
            payload = None
    return {
        "status": result.get("status"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "spearman_rho": None if payload is None else payload.get("spearman_rho"),
        "n_valid": None if payload is None else payload.get("n_valid"),
        "n_total": None if payload is None else payload.get("n_total"),
        "mean_tr": None if payload is None else payload.get("mean_tr"),
        "mean_cs": None if payload is None else payload.get("mean_cs"),
        "payload": payload,
    }


def run_cpu_night_campaign() -> Path:
    os.environ["ALLOW_CPU_SPACE"] = "1"
    from scripts.remote_worker import remote_worker_run

    profiles = ["short", "medium", "long"]
    started_at = datetime.now(UTC).isoformat()
    runs: list[dict[str, Any]] = []
    for profile in profiles:
        print(f"[CPUCampaign] profile={profile}")
        result = remote_worker_run(
            "run_qwen_per_case_diagnostic_v2.py",
            {
                "profile": profile,
                "max_new_tokens": 8,
                "group_case_cap": 1,
            },
            timeout=300,
        )
        summary = _extract_summary(result)
        runs.append(
            {
                "profile": profile,
                "request_args": {
                    "profile": profile,
                    "max_new_tokens": 8,
                    "group_case_cap": 1,
                },
                "worker_result": result,
                "summary": summary,
            }
        )
        print(
            f"[CPUCampaign] profile={profile} "
            f"status={summary['status']} elapsed={summary['elapsed_seconds']} "
            f"rho={summary['spearman_rho']}"
        )
        time.sleep(3)

    finished_at = datetime.now(UTC).isoformat()
    report = {
        "campaign": "cpu_night_campaign",
        "model": "Qwen/Qwen3.5-4B",
        "started_at": started_at,
        "finished_at": finished_at,
        "runs": runs,
    }
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = ARCHIVE_DIR / f"cpu_night_campaign_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CPUCampaign] saved_json={out_path}")
    return out_path


if __name__ == "__main__":
    run_cpu_night_campaign()
