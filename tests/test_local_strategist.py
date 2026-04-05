from __future__ import annotations

import json

from scripts import local_strategist


def test_parse_json_from_mixed_output_extracts_first_json_object() -> None:
    raw = "\n".join(
        [
            '{"template_id":"per_case_diag_short","reasoning":"pick the robust diagnostic"}',
            "OpenAI Codex v0.118.0",
            "user",
            "prompt",
            "codex",
            '{"template_id":"cross_profile_probe","reasoning":"secondary copy"}',
        ]
    )

    parsed = local_strategist._parse_json_from_mixed_output(raw)

    assert parsed == {
        "template_id": "per_case_diag_short",
        "reasoning": "pick the robust diagnostic",
    }


def test_codex_template_catalog_only_references_existing_scripts() -> None:
    catalog = local_strategist._build_codex_template_catalog()

    assert catalog
    available = set(local_strategist._list_run_qwen_scripts())
    for item in catalog:
        assert item["script"] in available
        assert item["template_id"]
        assert isinstance(item["args"], dict)


def test_build_backend_order_prefers_requested_backend(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGIST_FALLBACK_CHAIN", "codex,claude")

    assert local_strategist._build_backend_order("codex") == ["codex", "claude"]
    assert local_strategist._build_backend_order("claude") == ["claude", "codex"]


def test_local_strategist_falls_back_from_codex_to_claude(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGIST_BACKEND", "codex")
    monkeypatch.setenv("STRATEGIST_FALLBACK_CHAIN", "codex,claude")

    call_order: list[str] = []

    def fake_codex(*args, **kwargs):
        call_order.append("codex")
        return None, None

    def fake_claude(*args, **kwargs):
        call_order.append("claude")
        return {
            "id": "fallback_probe",
            "description": "Fallback proposal",
            "script": "run_qwen_phase_probe.py",
            "args": {"anchor_profile": "short"},
            "result_key": "correlation_summary.all_metrics.tail_retention_ratio",
            "success_threshold": 0.4,
            "reasoning": "Claude fallback after Codex failure",
            "script_code": "",
        }, json.dumps({"ok": True})

    monkeypatch.setattr(local_strategist, "_run_codex_backend", fake_codex)
    monkeypatch.setattr(local_strategist, "_run_claude_backend", fake_claude)

    proposal = local_strategist.strategist_local_select(
        state={"budget_remaining": 1, "current_phase": 1, "phases": {}, "known_facts": {}},
        playbook="",
        max_turns=1,
        per_call_timeout=1,
    )

    assert proposal is not None
    assert proposal["id"] == "fallback_probe"
    assert call_order == ["codex", "claude"]
