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


def detect_tree_domain(text: str, anchor_texts: Sequence[str] | None = None) -> str | None:
    haystack_parts = [text.lower()]
    if anchor_texts:
        haystack_parts.extend(anchor.lower() for anchor in anchor_texts)
    haystack = "\n".join(haystack_parts)

    math_hits = sum(marker in haystack for marker in _MATH_IBP_MARKERS)
    code_hits = sum(marker in haystack for marker in _CODE_FASTAPI_MARKERS)

    if math_hits == 0 and code_hits == 0:
        return None
    if math_hits >= code_hits:
        return "math_ibp"
    return "code_fastapi"

