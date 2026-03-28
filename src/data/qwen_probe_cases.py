from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenProbeCase:
    name: str
    description: str
    prompt: str
    expected_mode: str


def make_qwen_probe_cases() -> list[QwenProbeCase]:
    return [
        QwenProbeCase(
            name="quantifier_stable",
            description="Universal claim with internally consistent continuation pressure.",
            prompt=(
                "Claim: for all natural numbers n greater than zero, the statement is true. "
                "Continue in a way that preserves a universal mathematical reading."
            ),
            expected_mode="stable",
        ),
        QwenProbeCase(
            name="quantifier_conflict",
            description="Universal claim followed by pressure toward existential drift.",
            prompt=(
                "Claim: for all natural numbers n greater than zero, the statement is true. "
                "Then the text shifts toward saying there exists a witness instead. "
                "Continue the reasoning."
            ),
            expected_mode="conflict",
        ),
        QwenProbeCase(
            name="proof_mode_conflict",
            description="Contradiction-style proof prompt with later pressure toward direct proof.",
            prompt=(
                "We assume the negation and derive a contradiction. "
                "Halfway through, the text starts acting like a direct proof instead. "
                "Continue carefully."
            ),
            expected_mode="conflict",
        ),
        QwenProbeCase(
            name="induction_stable",
            description="Induction prompt with stable base-step-successor structure.",
            prompt=(
                "We prove the statement by induction: first the base case, then the induction step, "
                "then the successor case. Continue the proof in induction mode."
            ),
            expected_mode="stable",
        ),
        QwenProbeCase(
            name="api_framework_conflict",
            description="Framework prompt mixing conflicting implementation assumptions.",
            prompt=(
                "We are documenting a FastAPI service with async request handlers. "
                "Later the text starts describing it as a synchronous Django view layer. "
                "Continue the technical explanation."
            ),
            expected_mode="conflict",
        ),
    ]
