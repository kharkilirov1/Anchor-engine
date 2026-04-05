from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen_injection_geometry_probe.py"
SPEC = importlib.util.spec_from_file_location("run_qwen_injection_geometry_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_injected_prompt_contains_target_anchor_once() -> None:
    target = MODULE.QwenAnchorGeometryCase(
        name="target",
        anchor_class="content_like",
        anchor_group="json_only_response_format_policy",
        anchor_text="JSON only response format policy",
        prompt="Target prompt.",
        description="target",
    )
    host = MODULE.QwenAnchorGeometryCase(
        name="host",
        anchor_class="content_like",
        anchor_group="strictly_vegan_meal_plan_policy",
        anchor_text="strictly vegan meal plan policy",
        prompt="Host prompt.",
        description="host",
    )
    prompt = MODULE.build_injected_prompt(target, host)
    assert prompt.count(target.anchor_text) == 1
    assert "Host prompt." in prompt


def test_compute_auc_prefers_higher_positive_scores() -> None:
    auc = MODULE.compute_auc([0.1, 0.2, 0.3], [0.8, 0.9])
    assert auc == 1.0


def test_select_cases_respects_group_cap() -> None:
    case_a = MODULE.QwenAnchorGeometryCase("a", "content_like", "g1", "a1", "p1", "d1")
    case_b = MODULE.QwenAnchorGeometryCase("b", "content_like", "g1", "a2", "p2", "d2")
    case_c = MODULE.QwenAnchorGeometryCase("c", "content_like", "g2", "a3", "p3", "d3")
    selected = MODULE.select_cases([case_a, case_b, case_c], group_case_cap=1)
    assert [case.name for case in selected] == ["a", "c"]
