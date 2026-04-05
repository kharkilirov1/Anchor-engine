from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import remote_worker


def test_infer_worker_url_from_hf_remote(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WORKER_URL", raising=False)
    monkeypatch.setattr(remote_worker, "HF_SPACE_REPO", tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="https://huggingface.co/spaces/kharki/abpt\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    assert remote_worker._infer_worker_url() == "https://kharki-abpt.hf.space"


def test_remote_worker_refuses_cpu_only_space(monkeypatch) -> None:
    monkeypatch.delenv("ALLOW_CPU_SPACE", raising=False)
    monkeypatch.setattr(remote_worker, "_fetch_space_runtime", lambda: {
        "stage": "RUNNING",
        "hardware": {"current": "cpu-upgrade", "requested": "cpu-upgrade"},
    })
    monkeypatch.setattr(remote_worker, "_infer_worker_url", lambda: "https://kharki-abpt.hf.space")

    result = remote_worker.remote_worker_run("run_qwen_per_case_diagnostic_v2.py", {"profile": "short"})

    assert result["status"] == "error"
    assert "not on GPU hardware" in result["error"]


def test_remote_worker_returns_output_file_when_result_json_present(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ALLOW_CPU_SPACE", "1")
    monkeypatch.setattr(remote_worker, "_fetch_space_runtime", lambda: {
        "stage": "RUNNING",
        "hardware": {"current": "cpu-upgrade", "requested": "cpu-upgrade"},
    })
    monkeypatch.setattr(remote_worker, "_infer_worker_url", lambda: "https://kharki-abpt.hf.space")
    monkeypatch.setattr(remote_worker, "_sync_script_to_space", lambda script: None)

    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def predict(self, request_json: str, api_name: str) -> str:
            payload = json.loads(request_json)
            assert payload["script"] == "run_qwen_per_case_diagnostic_v2.py"
            return json.dumps({
                "status": "success",
                "elapsed_seconds": 12.3,
                "result_json": {"spearman_rho": 0.6, "n_valid": 5},
            })

    monkeypatch.setattr("gradio_client.Client", FakeClient)
    monkeypatch.setattr(
        remote_worker,
        "_save_result_locally",
        lambda script, result_json: str(tmp_path / f"{script}.json"),
    )

    result = remote_worker.remote_worker_run("run_qwen_per_case_diagnostic_v2.py", {"profile": "medium"})

    assert result["status"] == "success"
    assert result["output_file"] is not None
