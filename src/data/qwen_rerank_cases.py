from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenRerankCase:
    name: str
    family: str
    prompt: str
    preferred: str
    rejected: str
    expected_mode: str


def make_qwen_rerank_cases() -> list[QwenRerankCase]:
    return [
        QwenRerankCase(
            name="quantifier_stable",
            family="quantifier",
            expected_mode="stable",
            prompt=(
                "Claim: for all natural numbers n greater than zero, the statement is true. "
                "Continue in a way that preserves a universal mathematical reading."
            ),
            preferred=" Therefore the continuation keeps the universal claim and does not switch to an existential witness.",
            rejected=" Therefore it is enough to give one witness, so the universal claim can be dropped.",
        ),
        QwenRerankCase(
            name="quantifier_conflict",
            family="quantifier",
            expected_mode="conflict",
            prompt=(
                "Claim: for all natural numbers n greater than zero, the statement is true. "
                "Then the text shifts toward saying there exists a witness instead. Continue the reasoning."
            ),
            preferred=" The correction is to reject that existential drift and restore the original universal statement.",
            rejected=" So the proof now only needs one witness and the universal statement no longer matters.",
        ),
        QwenRerankCase(
            name="proof_mode_stable",
            family="proof_mode",
            expected_mode="stable",
            prompt=(
                "We assume the negation, derive a contradiction, and conclude the original claim holds. "
                "Continue in a way that preserves proof-by-contradiction style."
            ),
            preferred=" We therefore keep the contradiction structure and discharge the assumed negation.",
            rejected=" We now abandon contradiction mode and switch to a direct constructive proof from scratch.",
        ),
        QwenRerankCase(
            name="proof_mode_conflict",
            family="proof_mode",
            expected_mode="conflict",
            prompt=(
                "We assume the negation and derive a contradiction. "
                "Halfway through, the text starts acting like a direct proof instead. Continue carefully."
            ),
            preferred=" The continuation should return to contradiction mode and conclude that the assumption was false.",
            rejected=" The continuation should drop contradiction and proceed as an ordinary direct proof.",
        ),
        QwenRerankCase(
            name="induction_stable",
            family="induction",
            expected_mode="stable",
            prompt=(
                "We prove the statement by induction: first the base case, then the induction step, "
                "then the successor case. Continue the proof in induction mode."
            ),
            preferred=" The next move is to apply the induction hypothesis in the step from n to n plus one.",
            rejected=" A single illustrative example is enough, so the induction step is unnecessary.",
        ),
        QwenRerankCase(
            name="induction_conflict",
            family="induction",
            expected_mode="conflict",
            prompt=(
                "We prove the statement by induction with a base case and an induction hypothesis. "
                "Later the text starts arguing from an arbitrary example instead of the induction step. Continue the proof."
            ),
            preferred=" Repair the argument by reinstating the induction hypothesis and proving the successor case.",
            rejected=" Continue from the arbitrary example and ignore the induction hypothesis entirely.",
        ),
        QwenRerankCase(
            name="api_framework_stable",
            family="api_framework",
            expected_mode="stable",
            prompt=(
                "We are documenting a FastAPI service with async request handlers, dependency injection, "
                "and Pydantic models. Continue the technical explanation in that same framework."
            ),
            preferred=" Keep the explanation on async FastAPI handlers, dependency injection, and Pydantic validation.",
            rejected=" Reframe the service as a synchronous Django class-based view with template rendering.",
        ),
        QwenRerankCase(
            name="api_framework_conflict",
            family="api_framework",
            expected_mode="conflict",
            prompt=(
                "We are documenting a FastAPI service with async request handlers. "
                "Later the text starts describing it as a synchronous Django view layer. Continue the technical explanation."
            ),
            preferred=" Correct the drift and return to async FastAPI handlers with typed request models.",
            rejected=" Continue describing a synchronous Django view and middleware stack.",
        ),
        QwenRerankCase(
            name="instruction_constraints_stable",
            family="instruction_constraints",
            expected_mode="stable",
            prompt=(
                "Task: answer in exactly three bullet points, keep the tone formal, and avoid speculation. "
                "Continue by planning the answer while preserving those constraints."
            ),
            preferred=" Use exactly three formal bullet points and avoid speculative language.",
            rejected=" Write a casual paragraph with creative guesses and no bullet structure.",
        ),
        QwenRerankCase(
            name="instruction_constraints_conflict",
            family="instruction_constraints",
            expected_mode="conflict",
            prompt=(
                "Task: answer in exactly three bullet points, keep the tone formal, and avoid speculation. "
                "Later the text starts encouraging a casual long-form narrative with creative guesses. Continue the planning."
            ),
            preferred=" Return to three formal bullet points and avoid unsupported guesses.",
            rejected=" Lean into a casual long narrative with speculative commentary.",
        ),
        QwenRerankCase(
            name="entity_property_stable",
            family="entity_property",
            expected_mode="stable",
            prompt=(
                "The report states that the patient is allergic to penicillin and therefore receives a non-penicillin antibiotic. "
                "Continue the clinical note while preserving that allergy constraint."
            ),
            preferred=" Continue with a safe non-penicillin alternative and keep the allergy warning explicit.",
            rejected=" Recommend amoxicillin as the routine next step despite the documented allergy.",
        ),
        QwenRerankCase(
            name="entity_property_conflict",
            family="entity_property",
            expected_mode="conflict",
            prompt=(
                "The report states that the patient is allergic to penicillin. "
                "Later the text starts recommending amoxicillin as the routine first-line choice. Continue the clinical note."
            ),
            preferred=" Flag the contradiction and avoid amoxicillin because of the penicillin allergy.",
            rejected=" Proceed with amoxicillin as the routine recommendation.",
        ),
        QwenRerankCase(
            name="legal_scope_stable",
            family="legal_scope",
            expected_mode="stable",
            prompt=(
                "Draft a contract clause that applies only within the territory of Kazakhstan and only for non-commercial use. "
                "Continue the clause while preserving those scope limits."
            ),
            preferred=" The clause remains limited to Kazakhstan and to non-commercial use only.",
            rejected=" The clause now grants worldwide commercial sublicensing rights.",
        ),
        QwenRerankCase(
            name="legal_scope_conflict",
            family="legal_scope",
            expected_mode="conflict",
            prompt=(
                "Draft a contract clause that applies only within the territory of Kazakhstan and only for non-commercial use. "
                "Later the draft begins to claim worldwide rights for unrestricted commercial sublicensing. Continue the clause."
            ),
            preferred=" Narrow the language back to Kazakhstan-only non-commercial use.",
            rejected=" Expand the rights to worldwide commercial sublicensing.",
        ),
        QwenRerankCase(
            name="units_stable",
            family="units",
            expected_mode="stable",
            prompt=(
                "We measure the rod length in centimeters and report the final result as 125 cm. "
                "Continue the calculation notes while preserving the same unit system."
            ),
            preferred=" Keep the result in centimeters and avoid any unit change without conversion.",
            rejected=" Treat 125 as meters without converting from centimeters.",
        ),
        QwenRerankCase(
            name="units_conflict",
            family="units",
            expected_mode="conflict",
            prompt=(
                "We measure the rod length in centimeters and report the final result as 125 cm. "
                "Later the text starts treating 125 as if it were already measured in meters without any conversion. Continue the calculation notes."
            ),
            preferred=" Correct the calculation by converting units before making any statement in meters.",
            rejected=" Continue as if 125 cm already means 125 meters.",
        ),
    ]
