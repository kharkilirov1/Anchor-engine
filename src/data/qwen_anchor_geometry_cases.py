from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenAnchorGeometryCase:
    name: str
    anchor_class: str
    anchor_group: str
    anchor_text: str
    prompt: str
    description: str


def _case(
    *,
    name: str,
    anchor_class: str,
    anchor_group: str,
    anchor_text: str,
    prompt: str,
    description: str,
) -> QwenAnchorGeometryCase:
    return QwenAnchorGeometryCase(
        name=name,
        anchor_class=anchor_class,
        anchor_group=anchor_group,
        anchor_text=anchor_text,
        prompt=prompt,
        description=description,
    )


def make_qwen_anchor_geometry_cases() -> list[QwenAnchorGeometryCase]:
    return [
        _case(
            name="content_vegan_brief",
            anchor_class="content_like",
            anchor_group="strictly_vegan_meal_plan_policy",
            anchor_text="strictly vegan meal plan policy for every guest",
            prompt=(
                "The retreat brief requires a strictly vegan meal plan policy for every guest. "
                "Write a welcoming note explaining what guests can expect at meals, keeping the message consistent with a plant-based menu "
                "and avoiding dairy, eggs, and meat."
            ),
            description="Open-ended meal note anchored to a concrete vegan policy constraint.",
        ),
        _case(
            name="content_vegan_reason",
            anchor_class="content_like",
            anchor_group="strictly_vegan_meal_plan_policy",
            anchor_text="strictly vegan meal plan policy for every guest",
            prompt=(
                "The retreat brief requires a strictly vegan meal plan policy for every guest. "
                "Write a short explanation for attendees about why the meals stay plant-based and what substitutions they can expect instead "
                "of dairy or eggs."
            ),
            description="Open-ended rationale anchored to the same vegan policy.",
        ),
        _case(
            name="content_fastapi_architecture",
            anchor_class="content_like",
            anchor_group="async_fastapi_service_architecture_policy",
            anchor_text="async FastAPI service architecture policy for internal APIs",
            prompt=(
                "Our backend uses an async FastAPI service architecture policy for internal APIs. "
                "Write a short engineering note describing how requests move through the service, how dependencies are injected, "
                "and how Pydantic validation fits into the Python flow. Do not switch to Java, Spring, or SOAP terminology."
            ),
            description="Open-ended Python architecture note anchored to the FastAPI service policy.",
        ),
        _case(
            name="content_fastapi_summary",
            anchor_class="content_like",
            anchor_group="async_fastapi_service_architecture_policy",
            anchor_text="async FastAPI service architecture policy for internal APIs",
            prompt=(
                "Our backend uses an async FastAPI service architecture policy for internal APIs. "
                "Write a compact onboarding summary for a new teammate covering async request handling, dependency injection, "
                "and schema validation in Python."
            ),
            description="Open-ended onboarding summary anchored to the same FastAPI policy.",
        ),
        _case(
            name="content_json_contract",
            anchor_class="content_like",
            anchor_group="json_only_response_format_policy",
            anchor_text="JSON only response format policy for every endpoint",
            prompt=(
                "The integration contract enforces a JSON only response format policy for every endpoint. "
                "Write a short implementation note explaining what clients can expect from responses, how serialization should behave, "
                "and what parsers rely on. Keep it in API and JSON terms, not HTML or markdown."
            ),
            description="Open-ended API note anchored to the JSON response contract.",
        ),
        _case(
            name="content_json_parser",
            anchor_class="content_like",
            anchor_group="json_only_response_format_policy",
            anchor_text="JSON only response format policy for every endpoint",
            prompt=(
                "The integration contract enforces a JSON only response format policy for every endpoint. "
                "Write a brief note for client developers about why this helps downstream parsers and keeps schema handling predictable. "
                "Keep the discussion local to decoding and response structure."
            ),
            description="Open-ended parser-facing rationale anchored to the same JSON policy.",
        ),
        _case(
            name="procedure_contradiction_proof",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps for the claim",
            prompt=(
                "The proof outline says to use the proof by contradiction reasoning steps for the claim that if n^2 is even then n is even. "
                "Continue the outline from the negated claim to the contradiction, keeping the reasoning explicit and procedural."
            ),
            description="Open-ended contradiction proof anchored to a concrete classical claim.",
        ),
        _case(
            name="procedure_contradiction_explain",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps for the claim",
            prompt=(
                "The proof outline uses the proof by contradiction reasoning steps for the claim under discussion. "
                "Write a short explanation for a student about why the method begins by assuming the negated claim and why reaching a contradiction finishes the argument."
            ),
            description="Open-ended student explanation anchored to contradiction procedure.",
        ),
        _case(
            name="procedure_contradiction_surd_sum",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps for the claim",
            prompt=(
                "The proof outline uses the proof by contradiction reasoning steps for the claim that sqrt(2) + sqrt(3) is irrational. "
                "Continue the proof in an elementary-algebra style and make the contradiction explicit. "
                "Do not switch to numerical approximation."
            ),
            description="Open-ended harder contradiction proof anchored to an irrationality claim.",
        ),
        _case(
            name="procedure_binary_search_note",
            anchor_class="procedure_like",
            anchor_group="binary_search_update_loop_procedure",
            anchor_text="binary search update loop procedure on a sorted array",
            prompt=(
                "The algorithm note uses a binary search update loop procedure on a sorted array. "
                "Continue the note by describing how mid is computed, how it is compared to the target, and how the search interval shrinks. "
                "Do not switch to other algorithms."
            ),
            description="Open-ended algorithm note anchored to binary search updates.",
        ),
        _case(
            name="procedure_binary_search_indices",
            anchor_class="procedure_like",
            anchor_group="binary_search_update_loop_procedure",
            anchor_text="binary search update loop procedure on a sorted array",
            prompt=(
                "The algorithm note uses a binary search update loop procedure on a sorted array. "
                "Write a brief walkthrough of how low and high change after each comparison, including the cases target < arr[mid], "
                "target > arr[mid], and equality."
            ),
            description="Open-ended walkthrough anchored to precise binary search branch updates.",
        ),
        _case(
            name="procedure_di_request_path",
            anchor_class="procedure_like",
            anchor_group="dependency_injection_request_flow_sequence",
            anchor_text="dependency injection request flow sequence in a web service",
            prompt=(
                "The architecture note describes a dependency injection request flow sequence in a web service. "
                "Continue the note from request entry to handler execution. "
                "Mention container resolution, injected service construction, and handler call. "
                "Do not mention SOAP, Java, or reflection."
            ),
            description="Open-ended runtime flow note anchored to the DI request sequence.",
        ),
        _case(
            name="procedure_di_summary",
            anchor_class="procedure_like",
            anchor_group="dependency_injection_request_flow_sequence",
            anchor_text="dependency injection request flow sequence in a web service",
            prompt=(
                "The architecture note describes a dependency injection request flow sequence in a web service. "
                "Write a short onboarding summary from app startup to handler call, mentioning startup wiring, provider registration, "
                "request resolution, and handler invocation."
            ),
            description="Open-ended onboarding summary anchored to the same DI flow.",
        ),
    ]
