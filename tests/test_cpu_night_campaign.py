from __future__ import annotations

import json

from scripts import run_cpu_night_campaign


def test_extract_summary_from_final_result_stdout() -> None:
    result = {
        "status": "success",
        "elapsed_seconds": 123.4,
        "stdout_tail": "\n===FINAL_RESULT===\n" + json.dumps(
            {
                "spearman_rho": -0.3,
                "n_valid": 5,
                "n_total": 6,
                "mean_tr": 3.42,
                "mean_cs": 0.4,
            }
        ),
    }

    summary = run_cpu_night_campaign._extract_summary(result)

    assert summary["status"] == "success"
    assert summary["elapsed_seconds"] == 123.4
    assert summary["spearman_rho"] == -0.3
    assert summary["n_valid"] == 5
