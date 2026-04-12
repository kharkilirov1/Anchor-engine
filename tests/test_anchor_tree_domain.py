from __future__ import annotations

from src.model.anchor_tree_domain import detect_tree_domain


def test_detect_tree_domain_prefers_math_ibp() -> None:
    text = "Solve the integral using integration by parts only. Let u = x^2 and dv = e^x dx."

    assert detect_tree_domain(text) == "math_ibp"


def test_detect_tree_domain_prefers_fastapi() -> None:
    text = "Build an async FastAPI endpoint with Pydantic models and await the service call."

    assert detect_tree_domain(text) == "code_fastapi"


def test_detect_tree_domain_prefers_quantifier_scope() -> None:
    text = "Claim: for all natural numbers n the statement is true, so reject any existential witness drift."

    assert detect_tree_domain(text) == "quantifier"


def test_detect_tree_domain_prefers_proof_mode() -> None:
    text = "We assume the negation, derive a contradiction, and must not switch to a direct proof."

    assert detect_tree_domain(text) == "proof_mode"

