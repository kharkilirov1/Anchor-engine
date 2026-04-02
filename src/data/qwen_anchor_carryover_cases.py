from __future__ import annotations

from dataclasses import dataclass

from src.data.qwen_anchor_geometry_cases import ANCHOR_SPAN_PROFILES, ANCHOR_TEXT_BY_GROUP


@dataclass(frozen=True)
class QwenAnchorCarryoverCase:
    name: str
    anchor_group: str
    anchored_prefix: str
    neutral_prefix: str
    shared_suffix: str
    divergence_text: str


def _anchor_text_for_profile(anchor_group: str, anchor_span_profile: str) -> str:
    profile_map = ANCHOR_TEXT_BY_GROUP.get(anchor_group)
    if profile_map is None:
        raise KeyError(f"unknown anchor group: {anchor_group}")
    try:
        return profile_map[anchor_span_profile]
    except KeyError as exc:
        raise ValueError(
            f"unknown anchor span profile: {anchor_span_profile}; expected one of {ANCHOR_SPAN_PROFILES}"
        ) from exc


def make_qwen_anchor_carryover_cases(anchor_span_profile: str = "long") -> list[QwenAnchorCarryoverCase]:
    return [
        QwenAnchorCarryoverCase(
            name="carryover_vegan",
            anchor_group="strictly_vegan_meal_plan_policy",
            anchored_prefix=(
                "The retreat brief requires a "
                f"{_anchor_text_for_profile('strictly_vegan_meal_plan_policy', anchor_span_profile)}."
            ),
            neutral_prefix="The retreat brief includes a guest meal summary for every guest.",
            shared_suffix=" Write a welcoming note explaining what guests can expect at meals.",
            divergence_text="Write a welcoming note explaining what guests can expect at meals.",
        ),
        QwenAnchorCarryoverCase(
            name="carryover_fastapi",
            anchor_group="async_fastapi_service_architecture_policy",
            anchored_prefix=(
                "Our backend follows the "
                f"{_anchor_text_for_profile('async_fastapi_service_architecture_policy', anchor_span_profile)}."
            ),
            neutral_prefix="Our backend keeps a backend service summary for internal APIs.",
            shared_suffix=" Write a short engineering note describing request flow, validation, and dependency wiring in Python.",
            divergence_text="Write a short engineering note describing request flow, validation, and dependency wiring in Python.",
        ),
        QwenAnchorCarryoverCase(
            name="carryover_json",
            anchor_group="json_only_response_format_policy",
            anchored_prefix=(
                "The integration contract enforces a "
                f"{_anchor_text_for_profile('json_only_response_format_policy', anchor_span_profile)}."
            ),
            neutral_prefix="The integration contract includes a response structure summary for every endpoint.",
            shared_suffix=" Write a short implementation note about what clients can expect from responses and how parsers rely on structure.",
            divergence_text="Write a short implementation note about what clients can expect from responses and how parsers rely on structure.",
        ),
        QwenAnchorCarryoverCase(
            name="carryover_contradiction",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchored_prefix=(
                "The proof outline uses the "
                f"{_anchor_text_for_profile('proof_by_contradiction_reasoning_steps', anchor_span_profile)}"
                " under discussion."
            ),
            neutral_prefix="The proof handout contains a proof outline summary for the claim under discussion.",
            shared_suffix=" Write a short explanation for a student about how the argument proceeds from the setup to the conclusion.",
            divergence_text="Write a short explanation for a student about how the argument proceeds from the setup to the conclusion.",
        ),
        QwenAnchorCarryoverCase(
            name="carryover_binary_search",
            anchor_group="binary_search_update_loop_procedure",
            anchored_prefix=(
                "The algorithm note uses a "
                f"{_anchor_text_for_profile('binary_search_update_loop_procedure', anchor_span_profile)}."
            ),
            neutral_prefix="The algorithm handout includes a search routine summary on a sorted array.",
            shared_suffix=" Write a brief walkthrough of how low, high, and mid change after each comparison.",
            divergence_text="Write a brief walkthrough of how low, high, and mid change after each comparison.",
        ),
        QwenAnchorCarryoverCase(
            name="carryover_di",
            anchor_group="dependency_injection_request_flow_sequence",
            anchored_prefix=(
                "The architecture note describes a "
                f"{_anchor_text_for_profile('dependency_injection_request_flow_sequence', anchor_span_profile)}."
            ),
            neutral_prefix="The architecture note includes a request flow summary in a web service.",
            shared_suffix=" Write a short onboarding note about startup wiring, request resolution, and handler invocation.",
            divergence_text="Write a short onboarding note about startup wiring, request resolution, and handler invocation.",
        ),
    ]
