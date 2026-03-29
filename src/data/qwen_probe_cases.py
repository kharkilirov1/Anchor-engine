from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenProbeCase:
    name: str
    family: str
    description: str
    prompt: str
    expected_mode: str


def _case(
    name: str,
    family: str,
    description: str,
    prompt: str,
    expected_mode: str,
) -> QwenProbeCase:
    return QwenProbeCase(
        name=name,
        family=family,
        description=description,
        prompt=prompt,
        expected_mode=expected_mode,
    )


def make_qwen_probe_cases() -> list[QwenProbeCase]:
    return [
        _case(
            name="quantifier_stable",
            family="quantifier",
            description="Universal claim with internally consistent continuation pressure.",
            prompt=(
                "Claim: for all natural numbers n greater than zero, the statement is true. "
                "Continue in a way that preserves a universal mathematical reading."
            ),
            expected_mode="stable",
        ),
        _case(
            name="quantifier_conflict",
            family="quantifier",
            description="Universal claim followed by pressure toward existential drift.",
            prompt=(
                "Claim: for all natural numbers n greater than zero, the statement is true. "
                "Then the text shifts toward saying there exists a witness instead. "
                "Continue the reasoning."
            ),
            expected_mode="conflict",
        ),
        _case(
            name="proof_mode_stable",
            family="proof_mode",
            description="Proof by contradiction stays in contradiction mode throughout.",
            prompt=(
                "We assume the negation, derive a contradiction, and conclude the original claim holds. "
                "Continue in a way that preserves proof-by-contradiction style."
            ),
            expected_mode="stable",
        ),
        _case(
            name="proof_mode_conflict",
            family="proof_mode",
            description="Contradiction-style proof prompt with later pressure toward direct proof.",
            prompt=(
                "We assume the negation and derive a contradiction. "
                "Halfway through, the text starts acting like a direct proof instead. "
                "Continue carefully."
            ),
            expected_mode="conflict",
        ),
        _case(
            name="induction_stable",
            family="induction",
            description="Induction prompt with stable base-step-successor structure.",
            prompt=(
                "We prove the statement by induction: first the base case, then the induction step, "
                "then the successor case. Continue the proof in induction mode."
            ),
            expected_mode="stable",
        ),
        _case(
            name="induction_conflict",
            family="induction",
            description="Induction prompt that later slips into an ungrounded direct argument.",
            prompt=(
                "We prove the statement by induction with a base case and an induction hypothesis. "
                "Later the text starts arguing from an arbitrary example instead of the induction step. "
                "Continue the proof."
            ),
            expected_mode="conflict",
        ),
        _case(
            name="api_framework_stable",
            family="api_framework",
            description="FastAPI explanation remains internally aligned with async Python service assumptions.",
            prompt=(
                "We are documenting a FastAPI service with async request handlers, dependency injection, "
                "and Pydantic models. Continue the technical explanation in that same framework."
            ),
            expected_mode="stable",
        ),
        _case(
            name="api_framework_conflict",
            family="api_framework",
            description="Framework prompt mixing conflicting implementation assumptions.",
            prompt=(
                "We are documenting a FastAPI service with async request handlers. "
                "Later the text starts describing it as a synchronous Django view layer. "
                "Continue the technical explanation."
            ),
            expected_mode="conflict",
        ),
        _case(
            name="instruction_constraints_stable",
            family="instruction_constraints",
            description="Response instructions remain aligned with the original output constraints.",
            prompt=(
                "Task: answer in exactly three bullet points, keep the tone formal, and avoid speculation. "
                "Continue by planning the answer while preserving those constraints."
            ),
            expected_mode="stable",
        ),
        _case(
            name="instruction_constraints_conflict",
            family="instruction_constraints",
            description="Output instructions later push against the original response format and tone.",
            prompt=(
                "Task: answer in exactly three bullet points, keep the tone formal, and avoid speculation. "
                "Later the text starts encouraging a casual long-form narrative with creative guesses. "
                "Continue the planning."
            ),
            expected_mode="conflict",
        ),
        _case(
            name="entity_property_stable",
            family="entity_property",
            description="Entity attributes stay consistent across the prompt.",
            prompt=(
                "The report states that the patient is allergic to penicillin and therefore receives a non-penicillin antibiotic. "
                "Continue the clinical note while preserving that allergy constraint."
            ),
            expected_mode="stable",
        ),
        _case(
            name="entity_property_conflict",
            family="entity_property",
            description="A later sentence contradicts an earlier core entity attribute.",
            prompt=(
                "The report states that the patient is allergic to penicillin. "
                "Later the text starts recommending amoxicillin as the routine first-line choice. "
                "Continue the clinical note."
            ),
            expected_mode="conflict",
        ),
        _case(
            name="legal_scope_stable",
            family="legal_scope",
            description="Scope limitation remains consistent through the drafting prompt.",
            prompt=(
                "Draft a contract clause that applies only within the territory of Kazakhstan and only for non-commercial use. "
                "Continue the clause while preserving those scope limits."
            ),
            expected_mode="stable",
        ),
        _case(
            name="legal_scope_conflict",
            family="legal_scope",
            description="Territorial and usage scope drift into broader contradictory coverage.",
            prompt=(
                "Draft a contract clause that applies only within the territory of Kazakhstan and only for non-commercial use. "
                "Later the draft begins to claim worldwide rights for unrestricted commercial sublicensing. "
                "Continue the clause."
            ),
            expected_mode="conflict",
        ),
        _case(
            name="units_stable",
            family="units",
            description="A physics explanation keeps units and magnitude assumptions consistent.",
            prompt=(
                "We measure the rod length in centimeters and report the final result as 125 cm. "
                "Continue the calculation notes while preserving the same unit system."
            ),
            expected_mode="stable",
        ),
        _case(
            name="units_conflict",
            family="units",
            description="The prompt later mixes incompatible units without conversion.",
            prompt=(
                "We measure the rod length in centimeters and report the final result as 125 cm. "
                "Later the text starts treating 125 as if it were already measured in meters without any conversion. "
                "Continue the calculation notes."
            ),
            expected_mode="conflict",
        ),
    ]
