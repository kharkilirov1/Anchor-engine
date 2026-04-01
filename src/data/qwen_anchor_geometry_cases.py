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
            anchor_text="strictly vegan meal plan policy",
            prompt=(
                "The retreat brief requires a strictly vegan meal plan policy for every guest. "
                "Continue the short planning note in the same style."
            ),
            description="Food constraint phrased as a stable content policy.",
        ),
        _case(
            name="content_vegan_reason",
            anchor_class="content_like",
            anchor_group="strictly_vegan_meal_plan_policy",
            anchor_text="strictly vegan meal plan policy",
            prompt=(
                "Write one paragraph explaining why the strictly vegan meal plan policy excludes dairy and eggs. "
                "Keep the explanation concise."
            ),
            description="Same content anchor in an explanatory context.",
        ),
        _case(
            name="content_fastapi_architecture",
            anchor_class="content_like",
            anchor_group="async_fastapi_service_architecture_policy",
            anchor_text="async FastAPI service architecture policy",
            prompt=(
                "Our backend uses an async FastAPI service architecture policy for internal APIs. "
                "Continue the technical note with one more sentence."
            ),
            description="Service identity framed as a stable system property.",
        ),
        _case(
            name="content_fastapi_summary",
            anchor_class="content_like",
            anchor_group="async_fastapi_service_architecture_policy",
            anchor_text="async FastAPI service architecture policy",
            prompt=(
                "Summarize the async FastAPI service architecture policy for request handling and validation. "
                "Stay within the same technical frame."
            ),
            description="Same content anchor in a local documentation summary.",
        ),
        _case(
            name="content_json_contract",
            anchor_class="content_like",
            anchor_group="json_only_response_format_policy",
            anchor_text="JSON only response format policy",
            prompt=(
                "The integration contract enforces a JSON only response format policy for every endpoint. "
                "Continue the guideline in one short paragraph."
            ),
            description="Output format framed as a fixed content policy.",
        ),
        _case(
            name="content_json_parser",
            anchor_class="content_like",
            anchor_group="json_only_response_format_policy",
            anchor_text="JSON only response format policy",
            prompt=(
                "Explain why the JSON only response format policy helps downstream parsers and clients. "
                "Keep the wording technical and local."
            ),
            description="Same content anchor in an implementation rationale.",
        ),
        _case(
            name="procedure_contradiction_proof",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps",
            prompt=(
                "The proof outline says to use the proof by contradiction reasoning steps for this claim. "
                "Continue the proof sketch step by step."
            ),
            description="Proof procedure with explicit stepwise continuation pressure.",
        ),
        _case(
            name="procedure_contradiction_explain",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps",
            prompt=(
                "Explain why the proof by contradiction reasoning steps start from the negated claim. "
                "Keep the explanation short and procedural."
            ),
            description="Same procedure anchor in explanatory mode.",
        ),
        _case(
            name="procedure_binary_search_note",
            anchor_class="procedure_like",
            anchor_group="binary_search_update_loop_procedure",
            anchor_text="binary search update loop procedure",
            prompt=(
                "The algorithm note uses a binary search update loop procedure on a sorted array. "
                "Continue the explanation with the next local step."
            ),
            description="Algorithmic procedure with iterative state changes.",
        ),
        _case(
            name="procedure_binary_search_indices",
            anchor_class="procedure_like",
            anchor_group="binary_search_update_loop_procedure",
            anchor_text="binary search update loop procedure",
            prompt=(
                "Describe how the binary search update loop procedure changes low and high indices after each comparison. "
                "Keep the wording precise."
            ),
            description="Same procedure anchor focused on branch updates.",
        ),
        _case(
            name="procedure_di_request_path",
            anchor_class="procedure_like",
            anchor_group="dependency_injection_request_flow_sequence",
            anchor_text="dependency injection request flow sequence",
            prompt=(
                "The architecture note describes a dependency injection request flow sequence in the web service. "
                "Continue the explanation from request entry to handler execution."
            ),
            description="Framework procedure framed as a staged runtime flow.",
        ),
        _case(
            name="procedure_di_summary",
            anchor_class="procedure_like",
            anchor_group="dependency_injection_request_flow_sequence",
            anchor_text="dependency injection request flow sequence",
            prompt=(
                "Summarize the dependency injection request flow sequence from app startup to handler call. "
                "Use one compact paragraph."
            ),
            description="Same procedure anchor in a short runtime summary.",
        ),
    ]
