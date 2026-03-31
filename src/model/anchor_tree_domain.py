from __future__ import annotations

from collections.abc import Sequence

_MATH_IBP_MARKERS = (
    "integration by parts",
    "u =",
    "dv =",
    "integral",
    "∫",
)

_CODE_FASTAPI_MARKERS = (
    "fastapi",
    "pydantic",
    "async",
    "await",
    "request handler",
)

_QUANTIFIER_MARKERS = (
    "for all",
    "universal",
    "witness",
    "there exists",
    "existential",
)

_PROOF_MODE_MARKERS = (
    "contradiction",
    "assume the negation",
    "assumption was false",
    "direct proof",
    "constructive proof",
)


def detect_tree_domain(text: str, anchor_texts: Sequence[str] | None = None) -> str | None:
    haystack_parts = [text.lower()]
    if anchor_texts:
        haystack_parts.extend(anchor.lower() for anchor in anchor_texts)
    haystack = "\n".join(haystack_parts)

    scores = {
        "math_ibp": sum(marker in haystack for marker in _MATH_IBP_MARKERS),
        "code_fastapi": sum(marker in haystack for marker in _CODE_FASTAPI_MARKERS),
        "quantifier": sum(marker in haystack for marker in _QUANTIFIER_MARKERS),
        "proof_mode": sum(marker in haystack for marker in _PROOF_MODE_MARKERS),
    }
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return None
    return best_domain

