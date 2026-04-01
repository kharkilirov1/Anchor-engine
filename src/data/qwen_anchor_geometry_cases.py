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
                "Write exactly two policy sentences that ban dairy, eggs, and meat, and mention plant-based alternatives. "
                "Do not add dialogue, headers, or extra tasks."
            ),
            description="Food constraint phrased as a concrete local policy note.",
        ),
        _case(
            name="content_vegan_reason",
            anchor_class="content_like",
            anchor_group="strictly_vegan_meal_plan_policy",
            anchor_text="strictly vegan meal plan policy",
            prompt=(
                "Explain in exactly three sentences why the strictly vegan meal plan policy excludes dairy and eggs. "
                "Focus only on the policy rationale and plant-based substitutions."
            ),
            description="Same content anchor in a tightly scoped rationale.",
        ),
        _case(
            name="content_fastapi_architecture",
            anchor_class="content_like",
            anchor_group="async_fastapi_service_architecture_policy",
            anchor_text="async FastAPI service architecture policy",
            prompt=(
                "Our backend uses an async FastAPI service architecture policy for internal APIs. "
                "Write exactly three technical sentences about request flow, dependency injection, and Pydantic validation in Python. "
                "Do not mention Java, Spring, SOAP, or product management."
            ),
            description="Service identity framed as a tightly constrained Python note.",
        ),
        _case(
            name="content_fastapi_summary",
            anchor_class="content_like",
            anchor_group="async_fastapi_service_architecture_policy",
            anchor_text="async FastAPI service architecture policy",
            prompt=(
                "Summarize the async FastAPI service architecture policy in exactly three technical sentences. "
                "Stay in Python and FastAPI terms and mention async request handling plus schema validation."
            ),
            description="Same content anchor in a more explicit local documentation summary.",
        ),
        _case(
            name="content_json_contract",
            anchor_class="content_like",
            anchor_group="json_only_response_format_policy",
            anchor_text="JSON only response format policy",
            prompt=(
                "The integration contract enforces a JSON only response format policy for every endpoint. "
                "Write exactly three technical sentences explaining content type, serialization, and parser expectations. "
                "Do not mention HTML, markdown, or Java."
            ),
            description="Output format framed as a fixed content contract with explicit scope.",
        ),
        _case(
            name="content_json_parser",
            anchor_class="content_like",
            anchor_group="json_only_response_format_policy",
            anchor_text="JSON only response format policy",
            prompt=(
                "Explain in exactly three sentences why the JSON only response format policy helps downstream parsers and API clients. "
                "Keep the answer local to response decoding and schema consistency."
            ),
            description="Same content anchor in a constrained implementation rationale.",
        ),
        _case(
            name="procedure_contradiction_proof",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps",
            prompt=(
                "The proof outline says to use the proof by contradiction reasoning steps for the claim that if n^2 is even then n is even. "
                "Write exactly four short proof steps, starting from the negated claim and ending with the contradiction."
            ),
            description="Procedure anchor grounded in a concrete classical proof.",
        ),
        _case(
            name="procedure_contradiction_explain",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps",
            prompt=(
                "Explain in exactly three sentences why the proof by contradiction reasoning steps start from the negated claim. "
                "Keep the explanation procedural and mention the contradiction endpoint."
            ),
            description="Same procedure anchor in a constrained explanatory mode.",
        ),
        _case(
            name="procedure_contradiction_surd_sum",
            anchor_class="procedure_like",
            anchor_group="proof_by_contradiction_reasoning_steps",
            anchor_text="proof by contradiction reasoning steps",
            prompt=(
                "The proof outline uses the proof by contradiction reasoning steps for the claim that sqrt(2) + sqrt(3) is irrational. "
                "Write exactly five proof steps using only elementary algebra, and end with an explicit contradiction. "
                "Do not switch to numerical approximation."
            ),
            description="Harder concrete contradiction proof intended to stress a small model.",
        ),
        _case(
            name="procedure_binary_search_note",
            anchor_class="procedure_like",
            anchor_group="binary_search_update_loop_procedure",
            anchor_text="binary search update loop procedure",
            prompt=(
                "The algorithm note uses a binary search update loop procedure on a sorted array. "
                "Write exactly three steps describing how mid is computed, compared to the target, and how the search interval shrinks. "
                "Do not switch to other algorithms."
            ),
            description="Algorithmic procedure with explicit local update steps.",
        ),
        _case(
            name="procedure_binary_search_indices",
            anchor_class="procedure_like",
            anchor_group="binary_search_update_loop_procedure",
            anchor_text="binary search update loop procedure",
            prompt=(
                "Describe in exactly three sentences how the binary search update loop procedure changes low and high after each comparison. "
                "Mention the cases target < arr[mid], target > arr[mid], and equality."
            ),
            description="Same procedure anchor focused on the precise branch updates.",
        ),
        _case(
            name="procedure_di_request_path",
            anchor_class="procedure_like",
            anchor_group="dependency_injection_request_flow_sequence",
            anchor_text="dependency injection request flow sequence",
            prompt=(
                "The architecture note describes a dependency injection request flow sequence in a Python web service. "
                "Write exactly four steps from request entry to handler execution. "
                "Mention container resolution, injected service construction, and handler call. "
                "Do not mention SOAP, Java, or reflection."
            ),
            description="Framework procedure framed as a staged Python runtime flow.",
        ),
        _case(
            name="procedure_di_summary",
            anchor_class="procedure_like",
            anchor_group="dependency_injection_request_flow_sequence",
            anchor_text="dependency injection request flow sequence",
            prompt=(
                "Summarize the dependency injection request flow sequence from app startup to handler call in exactly four sentences. "
                "Stay in Python web-service terms and mention startup wiring, provider registration, request resolution, and handler invocation."
            ),
            description="Same procedure anchor in a compact but explicit runtime summary.",
        ),
    ]
