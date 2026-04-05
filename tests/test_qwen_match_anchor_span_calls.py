from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _match_calls(path: Path) -> list[ast.Call]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "match_anchor_span":
            calls.append(node)
    return calls


def test_match_anchor_span_calls_use_keyword_arguments() -> None:
    targets = [
        ROOT / "scripts" / "run_qwen_cross_profile_probe.py",
        ROOT / "scripts" / "run_qwen_per_case_diagnostic.py",
        ROOT / "scripts" / "run_qwen_per_case_diagnostic_v2.py",
    ]

    for path in targets:
        calls = _match_calls(path)
        assert calls, f"no match_anchor_span call found in {path}"
        for call in calls:
            assert call.args == [], f"positional args regression in {path}"
            keyword_names = {kw.arg for kw in call.keywords}
            assert {"text", "anchor_text", "input_ids", "tokenizer", "offsets"} <= keyword_names


def test_diagnostic_scripts_do_not_use_removed_numpy_trapz() -> None:
    targets = [
        ROOT / "scripts" / "run_qwen_per_case_diagnostic.py",
        ROOT / "scripts" / "run_qwen_per_case_diagnostic_v2.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "np.trapz(" not in text, f"deprecated np.trapz usage remains in {path}"
        assert "np.trapezoid(" in text, f"expected np.trapezoid compatibility fix missing in {path}"
