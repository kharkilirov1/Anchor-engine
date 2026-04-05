from __future__ import annotations

import json

from scripts import run_cpu_large_campaign


def test_extract_summary_from_large_campaign_stdout() -> None:
    result = {
        "status": "success",
        "elapsed_seconds": 140.0,
        "stdout_tail": "\n===FINAL_RESULT===\n" + json.dumps(
            {
                "spearman_rho": 0.6,
                "n_valid": 6,
                "n_total": 6,
                "mean_tr": 3.31,
                "mean_cs": 0.33,
            }
        ),
    }

    summary = run_cpu_large_campaign._extract_summary(result)

    assert summary["status"] == "success"
    assert summary["spearman_rho"] == 0.6
    assert summary["n_valid"] == 6
