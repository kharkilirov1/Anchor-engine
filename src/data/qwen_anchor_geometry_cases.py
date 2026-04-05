from __future__ import annotations

from dataclasses import dataclass


ANCHOR_SPAN_PROFILES: tuple[str, ...] = ("short", "medium", "long")


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


ANCHOR_TEXT_BY_GROUP: dict[str, dict[str, str]] = {
    # --- existing groups ---
    "strictly_vegan_meal_plan_policy": {
        "short": "vegan meal plan policy",
        "medium": "strictly vegan meal plan policy",
        "long": "strictly vegan meal plan policy for every guest",
    },
    "async_fastapi_service_architecture_policy": {
        "short": "FastAPI service architecture policy",
        "medium": "async FastAPI service architecture policy",
        "long": "async FastAPI service architecture policy for internal APIs",
    },
    "json_only_response_format_policy": {
        "short": "JSON only response format",
        "medium": "JSON only response format policy",
        "long": "JSON only response format policy for every endpoint",
    },
    "proof_by_contradiction_reasoning_steps": {
        "short": "contradiction reasoning steps",
        "medium": "proof by contradiction reasoning steps",
        "long": "proof by contradiction reasoning steps for the claim",
    },
    "binary_search_update_loop_procedure": {
        "short": "search update loop procedure",
        "medium": "binary search update loop procedure",
        "long": "binary search update loop procedure on a sorted array",
    },
    "dependency_injection_request_flow_sequence": {
        "short": "injection request flow sequence",
        "medium": "dependency injection request flow sequence",
        "long": "dependency injection request flow sequence in a web service",
    },
    # --- new groups: rare/ambiguous/cross-domain ---
    "penicillin_allergy_treatment_protocol": {
        "short": "allergy treatment protocol",
        "medium": "penicillin allergy treatment protocol",
        "long": "penicillin allergy treatment protocol for the patient",
    },
    "gdpr_data_retention_compliance_policy": {
        "short": "data retention compliance policy",
        "medium": "GDPR data retention compliance policy",
        "long": "GDPR data retention compliance policy for all user records",
    },
    "mathematical_induction_proof_steps": {
        "short": "induction proof steps",
        "medium": "mathematical induction proof steps",
        "long": "mathematical induction proof steps for the inequality",
    },
    "sql_foreign_key_constraint_enforcement": {
        "short": "foreign key constraint enforcement",
        "medium": "SQL foreign key constraint enforcement",
        "long": "SQL foreign key constraint enforcement on every table",
    },
    "thread_safe_singleton_initialization_pattern": {
        "short": "singleton initialization pattern",
        "medium": "thread-safe singleton initialization pattern",
        "long": "thread-safe singleton initialization pattern with double-checked locking",
    },
    "idempotent_rest_api_retry_policy": {
        "short": "API retry policy",
        "medium": "idempotent REST API retry policy",
        "long": "idempotent REST API retry policy for every write endpoint",
    },
    "recursive_tree_traversal_procedure": {
        "short": "tree traversal procedure",
        "medium": "recursive tree traversal procedure",
        "long": "recursive tree traversal procedure on a binary tree",
    },
    "strict_typescript_null_safety_policy": {
        "short": "null safety policy",
        "medium": "strict TypeScript null safety policy",
        "long": "strict TypeScript null safety policy across the codebase",
    },
}


def list_anchor_span_profiles() -> tuple[str, ...]:
    return ANCHOR_SPAN_PROFILES


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


def make_qwen_anchor_geometry_cases(
    anchor_span_profile: str = "long",
) -> list[QwenAnchorGeometryCase]:
    return [
        _case(
            name="content_vegan_brief",
            anchor_class="content_like",
            anchor_group="strictly_vegan_meal_plan_policy",
            anchor_text=_anchor_text_for_profile(
                "strictly_vegan_meal_plan_policy",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "strictly_vegan_meal_plan_policy",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "async_fastapi_service_architecture_policy",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "async_fastapi_service_architecture_policy",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "json_only_response_format_policy",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "json_only_response_format_policy",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "proof_by_contradiction_reasoning_steps",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "proof_by_contradiction_reasoning_steps",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "proof_by_contradiction_reasoning_steps",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "binary_search_update_loop_procedure",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "binary_search_update_loop_procedure",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "dependency_injection_request_flow_sequence",
                anchor_span_profile,
            ),
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
            anchor_text=_anchor_text_for_profile(
                "dependency_injection_request_flow_sequence",
                anchor_span_profile,
            ),
            prompt=(
                "The architecture note describes a dependency injection request flow sequence in a web service. "
                "Write a short onboarding summary from app startup to handler call, mentioning startup wiring, provider registration, "
                "request resolution, and handler invocation."
            ),
            description="Open-ended onboarding summary anchored to the same DI flow.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: medical domain (rare, high-constraint, likely flat)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="content_allergy_prescription",
            anchor_class="content_like",
            anchor_group="penicillin_allergy_treatment_protocol",
            anchor_text=_anchor_text_for_profile(
                "penicillin_allergy_treatment_protocol",
                anchor_span_profile,
            ),
            prompt=(
                "The chart specifies a penicillin allergy treatment protocol for the patient. "
                "Write a brief clinical note listing safe antibiotic alternatives, "
                "emphasizing that penicillin and amoxicillin must be avoided. "
                "Do not recommend any beta-lactam antibiotics."
            ),
            description="Medical constraint: avoid penicillin-class drugs.",
        ),
        _case(
            name="content_allergy_discharge",
            anchor_class="content_like",
            anchor_group="penicillin_allergy_treatment_protocol",
            anchor_text=_anchor_text_for_profile(
                "penicillin_allergy_treatment_protocol",
                anchor_span_profile,
            ),
            prompt=(
                "The chart specifies a penicillin allergy treatment protocol for the patient. "
                "Write discharge instructions for the patient explaining which medications to avoid "
                "and what to tell future healthcare providers about the allergy."
            ),
            description="Patient-facing discharge note under penicillin allergy constraint.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: legal/GDPR domain (rare, content-like, likely flat)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="content_gdpr_retention",
            anchor_class="content_like",
            anchor_group="gdpr_data_retention_compliance_policy",
            anchor_text=_anchor_text_for_profile(
                "gdpr_data_retention_compliance_policy",
                anchor_span_profile,
            ),
            prompt=(
                "The system enforces a GDPR data retention compliance policy for all user records. "
                "Write a technical note for the engineering team explaining how personal data must be "
                "deleted or anonymized after the retention period, and what audit logs must be kept."
            ),
            description="GDPR data retention constraint for engineering audience.",
        ),
        _case(
            name="content_gdpr_user_notice",
            anchor_class="content_like",
            anchor_group="gdpr_data_retention_compliance_policy",
            anchor_text=_anchor_text_for_profile(
                "gdpr_data_retention_compliance_policy",
                anchor_span_profile,
            ),
            prompt=(
                "The system enforces a GDPR data retention compliance policy for all user records. "
                "Write a user-facing privacy notice explaining how long their data is stored, "
                "their right to erasure, and how to request deletion. Keep it plain language."
            ),
            description="User-facing GDPR notice under data retention policy.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: mathematical induction (procedure, tests math beyond contradiction)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="procedure_induction_sum",
            anchor_class="procedure_like",
            anchor_group="mathematical_induction_proof_steps",
            anchor_text=_anchor_text_for_profile(
                "mathematical_induction_proof_steps",
                anchor_span_profile,
            ),
            prompt=(
                "The proof uses mathematical induction proof steps for the inequality "
                "1 + 2 + ... + n = n(n+1)/2. "
                "Write the base case, inductive hypothesis, and inductive step explicitly. "
                "Do not use any method other than induction."
            ),
            description="Standard induction proof anchored to sum formula.",
        ),
        _case(
            name="procedure_induction_inequality",
            anchor_class="procedure_like",
            anchor_group="mathematical_induction_proof_steps",
            anchor_text=_anchor_text_for_profile(
                "mathematical_induction_proof_steps",
                anchor_span_profile,
            ),
            prompt=(
                "The proof uses mathematical induction proof steps for the inequality "
                "2^n > n for all n >= 1. "
                "State the base case, assume P(k), and derive P(k+1). "
                "Keep the argument self-contained and do not switch to a different proof technique."
            ),
            description="Induction proof for exponential inequality.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: SQL constraints (content-like, cross-domain DB+code)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="content_sql_fk_design",
            anchor_class="content_like",
            anchor_group="sql_foreign_key_constraint_enforcement",
            anchor_text=_anchor_text_for_profile(
                "sql_foreign_key_constraint_enforcement",
                anchor_span_profile,
            ),
            prompt=(
                "The database schema requires SQL foreign key constraint enforcement on every table. "
                "Write a design note explaining how referential integrity is maintained, "
                "what happens on DELETE CASCADE vs RESTRICT, and why orphan rows must be prevented."
            ),
            description="DB design note anchored to FK constraint enforcement.",
        ),
        _case(
            name="content_sql_fk_migration",
            anchor_class="content_like",
            anchor_group="sql_foreign_key_constraint_enforcement",
            anchor_text=_anchor_text_for_profile(
                "sql_foreign_key_constraint_enforcement",
                anchor_span_profile,
            ),
            prompt=(
                "The database schema requires SQL foreign key constraint enforcement on every table. "
                "Write a migration guide for adding foreign keys to an existing schema with legacy data. "
                "Cover how to handle orphan rows before enabling constraints."
            ),
            description="Migration guide under FK constraint policy.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: thread safety (procedure, concurrency domain)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="procedure_singleton_impl",
            anchor_class="procedure_like",
            anchor_group="thread_safe_singleton_initialization_pattern",
            anchor_text=_anchor_text_for_profile(
                "thread_safe_singleton_initialization_pattern",
                anchor_span_profile,
            ),
            prompt=(
                "The design document specifies a thread-safe singleton initialization pattern "
                "with double-checked locking. "
                "Write a step-by-step explanation of why naive singleton fails under concurrency, "
                "how double-checked locking solves it, and what role volatile/memory barriers play. "
                "Do not switch to dependency injection or static initialization."
            ),
            description="Thread-safe singleton procedure with concurrency focus.",
        ),
        _case(
            name="procedure_singleton_test",
            anchor_class="procedure_like",
            anchor_group="thread_safe_singleton_initialization_pattern",
            anchor_text=_anchor_text_for_profile(
                "thread_safe_singleton_initialization_pattern",
                anchor_span_profile,
            ),
            prompt=(
                "The design document specifies a thread-safe singleton initialization pattern "
                "with double-checked locking. "
                "Write a short note on how to test this pattern: what race conditions to simulate, "
                "how many threads to spawn, and what assertions verify correctness."
            ),
            description="Testing strategy for thread-safe singleton.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: REST idempotency (content, API domain)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="content_idempotent_design",
            anchor_class="content_like",
            anchor_group="idempotent_rest_api_retry_policy",
            anchor_text=_anchor_text_for_profile(
                "idempotent_rest_api_retry_policy",
                anchor_span_profile,
            ),
            prompt=(
                "The API specification mandates an idempotent REST API retry policy for every write endpoint. "
                "Write a design note explaining idempotency keys, how retries are safe under this policy, "
                "and what happens when a duplicate request arrives. "
                "Do not describe non-idempotent fire-and-forget patterns."
            ),
            description="Idempotent API design note.",
        ),
        _case(
            name="content_idempotent_client",
            anchor_class="content_like",
            anchor_group="idempotent_rest_api_retry_policy",
            anchor_text=_anchor_text_for_profile(
                "idempotent_rest_api_retry_policy",
                anchor_span_profile,
            ),
            prompt=(
                "The API specification mandates an idempotent REST API retry policy for every write endpoint. "
                "Write a client integration guide explaining how to generate idempotency keys, "
                "when to retry, and how to handle 409 Conflict responses."
            ),
            description="Client-facing idempotency guide.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: recursive traversal (procedure, algorithmic)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="procedure_recursive_inorder",
            anchor_class="procedure_like",
            anchor_group="recursive_tree_traversal_procedure",
            anchor_text=_anchor_text_for_profile(
                "recursive_tree_traversal_procedure",
                anchor_span_profile,
            ),
            prompt=(
                "The algorithm note uses a recursive tree traversal procedure on a binary tree. "
                "Describe the in-order traversal step by step: when the left subtree is visited, "
                "when the node value is recorded, and when the right subtree is visited. "
                "Do not switch to iterative traversal with an explicit stack."
            ),
            description="Recursive in-order traversal procedure.",
        ),
        _case(
            name="procedure_recursive_depth",
            anchor_class="procedure_like",
            anchor_group="recursive_tree_traversal_procedure",
            anchor_text=_anchor_text_for_profile(
                "recursive_tree_traversal_procedure",
                anchor_span_profile,
            ),
            prompt=(
                "The algorithm note uses a recursive tree traversal procedure on a binary tree. "
                "Write a brief note on computing tree depth recursively: "
                "what the base case returns, how left and right depths combine, "
                "and why the recursion terminates."
            ),
            description="Recursive depth computation on binary tree.",
        ),
        # ─────────────────────────────────────────────────────────────────────
        # NEW: TypeScript strict null (content, type-system domain)
        # ─────────────────────────────────────────────────────────────────────
        _case(
            name="content_ts_null_safety",
            anchor_class="content_like",
            anchor_group="strict_typescript_null_safety_policy",
            anchor_text=_anchor_text_for_profile(
                "strict_typescript_null_safety_policy",
                anchor_span_profile,
            ),
            prompt=(
                "The project enforces a strict TypeScript null safety policy across the codebase. "
                "Write a developer guide explaining strictNullChecks, how to handle optional values "
                "with narrowing and assertion functions, and why `any` and `!` operator are banned. "
                "Keep examples in TypeScript only."
            ),
            description="TypeScript null safety developer guide.",
        ),
        _case(
            name="content_ts_null_review",
            anchor_class="content_like",
            anchor_group="strict_typescript_null_safety_policy",
            anchor_text=_anchor_text_for_profile(
                "strict_typescript_null_safety_policy",
                anchor_span_profile,
            ),
            prompt=(
                "The project enforces a strict TypeScript null safety policy across the codebase. "
                "Write a code review checklist for catching null safety violations: "
                "what patterns to flag, how to suggest fixes, and when exceptions are acceptable."
            ),
            description="Code review checklist under null safety policy.",
        ),
    ]
