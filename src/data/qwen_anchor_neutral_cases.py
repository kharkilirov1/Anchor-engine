from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenAnchorNeutralCase:
    name: str
    domain: str
    focus_text: str
    prompt: str


def make_qwen_anchor_neutral_cases() -> list[QwenAnchorNeutralCase]:
    return [
        QwenAnchorNeutralCase(
            name="neutral_meal_note",
            domain="strictly_vegan_meal_plan_policy",
            focus_text="guest meal summary",
            prompt=(
                "The retreat brief includes a guest meal summary for every guest. "
                "Write a welcoming note explaining what guests can expect at meals."
            ),
        ),
        QwenAnchorNeutralCase(
            name="neutral_backend_note",
            domain="async_fastapi_service_architecture_policy",
            focus_text="backend service summary",
            prompt=(
                "Our backend keeps a backend service summary for internal APIs. "
                "Write a short engineering note about request flow and validation."
            ),
        ),
        QwenAnchorNeutralCase(
            name="neutral_response_note",
            domain="json_only_response_format_policy",
            focus_text="response structure summary",
            prompt=(
                "The integration contract includes a response structure summary for every endpoint. "
                "Write a short note about what clients can expect from responses."
            ),
        ),
        QwenAnchorNeutralCase(
            name="neutral_proof_note",
            domain="proof_by_contradiction_reasoning_steps",
            focus_text="proof outline summary",
            prompt=(
                "The proof handout contains a proof outline summary for the claim under discussion. "
                "Write a short explanation of how the argument proceeds."
            ),
        ),
        QwenAnchorNeutralCase(
            name="neutral_search_note",
            domain="binary_search_update_loop_procedure",
            focus_text="search routine summary",
            prompt=(
                "The algorithm handout includes a search routine summary on a sorted array. "
                "Write a brief walkthrough of how the interval changes."
            ),
        ),
        QwenAnchorNeutralCase(
            name="neutral_request_flow",
            domain="dependency_injection_request_flow_sequence",
            focus_text="request flow summary",
            prompt=(
                "The architecture note includes a request flow summary in a web service. "
                "Write a short onboarding note about startup wiring and request handling."
            ),
        ),
    ]
