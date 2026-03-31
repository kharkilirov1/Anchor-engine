from __future__ import annotations

from typing import Any

from src.model.anchor_tree import make_anchor_tree
from src.model.anchor_tree_domain import detect_tree_domain
from src.model.anchor_tree_types import AnchorTree, AnchorTreeEdge, AnchorTreeNode, AnchorTreeRelation, AnchorTreeRole
from src.model.anchor_types import AnchorRecord

_MATH_STEP_ORDER = {
    "integration_by_parts_only": 0,
    "select_u_and_dv": 1,
    "derive_du_and_v": 2,
    "substitute_uv_minus_int_vdu": 3,
    "reduce_integral_complexity": 4,
    "repeat_if_needed": 5,
    "simplify_result": 6,
    "integration_constant": 7,
    "shortcut_lookup": 50,
    "table_reference": 51,
    "substitution_switch": 52,
    "meta_abort": 53,
    "wrong_symbolic_step": 54,
}

_CODE_STEP_ORDER = {
    "async_fastapi_service": 0,
    "typed_request_models": 1,
    "dependency_injection": 2,
    "async_handlers": 3,
    "validation_path": 4,
    "background_jobs": 5,
    "deployment_notes": 6,
    "django_view_reframe": 50,
    "synchronous_handler_reframe": 51,
    "template_rendering_branch": 52,
}


def _domain_root_label(domain: str | None) -> str:
    if domain == "math_ibp":
        return "integration_by_parts_only"
    if domain == "code_fastapi":
        return "async_fastapi_service"
    return "observed_root"


def _step_order(domain: str | None) -> dict[str, int]:
    if domain == "math_ibp":
        return _MATH_STEP_ORDER
    if domain == "code_fastapi":
        return _CODE_STEP_ORDER
    return {}


def _classify_math_label(text: str) -> str:
    lowered = text.lower()
    if "integration by parts" in lowered:
        return "integration_by_parts_only"
    if ("let u" in lowered or "u =" in lowered) and "dv" in lowered:
        return "select_u_and_dv"
    if "du" in lowered and ("v =" in lowered or " v " in f" {lowered} "):
        return "derive_du_and_v"
    if "uv" in lowered or "vdu" in lowered or "substitut" in lowered:
        return "substitute_uv_minus_int_vdu"
    if "remaining integral" in lowered or "reduce" in lowered:
        return "reduce_integral_complexity"
    if "repeat" in lowered or "again" in lowered:
        return "repeat_if_needed"
    if "+ c" in lowered or "+c" in lowered or "constant of integration" in lowered:
        return "integration_constant"
    if "simplif" in lowered:
        return "simplify_result"
    if "shortcut" in lowered:
        return "shortcut_lookup"
    if "table" in lowered:
        return "table_reference"
    if "substitution" in lowered:
        return "substitution_switch"
    if any(marker in lowered for marker in ("too hard", "challenging", "no clear path", "alternative approach")):
        return "meta_abort"
    return "math_observed_step"


def _classify_code_label(text: str) -> str:
    lowered = text.lower()
    if "fastapi" in lowered and "async" in lowered:
        return "async_fastapi_service"
    if "pydantic" in lowered or "request model" in lowered or "response model" in lowered:
        return "typed_request_models"
    if "dependency injection" in lowered or "depends(" in lowered:
        return "dependency_injection"
    if "async handler" in lowered or ("async def" in lowered and "await" in lowered):
        return "async_handlers"
    if "validation" in lowered or "validate" in lowered:
        return "validation_path"
    if "background task" in lowered or "background job" in lowered:
        return "background_jobs"
    if "deploy" in lowered or "uvicorn" in lowered or "gunicorn" in lowered:
        return "deployment_notes"
    if "django" in lowered:
        return "django_view_reframe"
    if "synchronous" in lowered or "sync view" in lowered:
        return "synchronous_handler_reframe"
    if "template" in lowered or "render" in lowered:
        return "template_rendering_branch"
    return "code_observed_step"


def classify_observed_label(domain: str | None, text: str) -> str:
    if domain == "math_ibp":
        return _classify_math_label(text)
    if domain == "code_fastapi":
        return _classify_code_label(text)
    return "observed_step"


def _role_for_label(label: str, source: str, is_root: bool) -> AnchorTreeRole:
    if is_root:
        return AnchorTreeRole.CONSTRAINT
    if label in {
        "shortcut_lookup",
        "table_reference",
        "substitution_switch",
        "wrong_symbolic_step",
        "django_view_reframe",
        "synchronous_handler_reframe",
        "template_rendering_branch",
    }:
        return AnchorTreeRole.DRIFT
    if label == "meta_abort":
        return AnchorTreeRole.META
    if source == "auxiliary_proposal":
        return AnchorTreeRole.REPAIR
    if source == "future_hint":
        return AnchorTreeRole.DERIVED
    return AnchorTreeRole.STEP


def _score_anchor(anchor: AnchorRecord) -> float:
    support = float(anchor.support.detach().item()) if hasattr(anchor.support, "detach") else float(anchor.support)
    viability = float(anchor.viability.detach().item()) if hasattr(anchor.viability, "detach") else float(anchor.viability)
    return support * max(viability, 1e-6)


def _make_root_node(domain: str | None, text: str, root_anchor: AnchorRecord | None) -> AnchorTreeNode:
    root_score = _score_anchor(root_anchor) if root_anchor is not None else 0.0
    root_repr = root_anchor.repr.detach().clone() if root_anchor is not None else None
    root_start = int(root_anchor.start_idx) if root_anchor is not None else 0
    root_end = int(root_anchor.end_idx) if root_anchor is not None else max(0, len(text.split()) - 1)
    return AnchorTreeNode(
        node_id="root",
        label=_domain_root_label(domain),
        text=text,
        depth=0,
        role=AnchorTreeRole.CONSTRAINT,
        source="prompt",
        anchor_id=None if root_anchor is None else int(root_anchor.id),
        span_start=root_start,
        span_end=root_end,
        repr=root_repr,
        score=float(root_score),
    )


def _make_anchor_payloads(active_anchors: list[dict[str, Any] | AnchorRecord]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in active_anchors:
        if isinstance(item, AnchorRecord):
            payloads.append({
                "anchor": item,
                "text": f"anchor_{item.id}",
                "start": int(item.start_idx),
                "end": int(item.end_idx),
            })
        else:
            payloads.append(item)
    payloads.sort(key=lambda payload: (int(payload.get("start", 0)), int(payload.get("end", 0))))
    return payloads


def build_observed_tree(
    *,
    text: str,
    active_anchors: list[dict[str, Any] | AnchorRecord],
    future_hint_candidates: list[dict[str, Any]],
    auxiliary_proposals: list[dict[str, Any]],
    domain: str | None = None,
) -> AnchorTree:
    anchor_payloads = _make_anchor_payloads(active_anchors)
    anchor_texts = [str(payload.get("text", "")) for payload in anchor_payloads]
    resolved_domain = domain or detect_tree_domain(text=text, anchor_texts=anchor_texts)
    root_anchor = None
    if anchor_payloads:
        root_anchor = max(
            (payload["anchor"] for payload in anchor_payloads if isinstance(payload.get("anchor"), AnchorRecord)),
            key=_score_anchor,
            default=None,
        )
    root = _make_root_node(resolved_domain, text, root_anchor)

    nodes: list[AnchorTreeNode] = []
    edges: list[AnchorTreeEdge] = []
    order_map = _step_order(resolved_domain)

    normalized_items: list[dict[str, Any]] = []
    for idx, payload in enumerate(anchor_payloads):
        anchor = payload.get("anchor")
        payload_text = str(payload.get("text", "")).strip()
        node_label = classify_observed_label(resolved_domain, payload_text)
        if payload_text and node_label != root.label:
            normalized_items.append(
                {
                    "node_id": f"anchor_{idx}",
                    "label": node_label,
                    "text": payload_text,
                    "span_start": int(payload.get("start", 0)),
                    "span_end": int(payload.get("end", 0)),
                    "repr": None if anchor is None else anchor.repr.detach().clone(),
                    "score": 0.0 if anchor is None else _score_anchor(anchor),
                    "source": "active_anchor",
                }
            )

    for idx, hint in enumerate(future_hint_candidates):
        hint_text = str(hint.get("text", "")).strip()
        if not hint_text:
            continue
        normalized_items.append(
            {
                "node_id": f"hint_{idx}",
                "label": classify_observed_label(resolved_domain, hint_text),
                "text": hint_text,
                "span_start": int(hint.get("start", 0)),
                "span_end": int(hint.get("end", hint.get("start", 0))),
                "repr": None,
                "score": float(hint.get("mean_score", 0.0)),
                "source": "future_hint",
            }
        )

    for idx, proposal in enumerate(auxiliary_proposals):
        proposal_text = str(proposal.get("proposal_text", "")).strip()
        start, end = proposal.get("proposal_span", (0, 0))
        if not proposal_text:
            continue
        normalized_items.append(
            {
                "node_id": f"proposal_{idx}",
                "label": classify_observed_label(resolved_domain, proposal_text),
                "text": proposal_text,
                "span_start": int(start),
                "span_end": int(end),
                "repr": proposal.get("repr"),
                "score": float(proposal.get("proposal_score", 0.0)),
                "source": "auxiliary_proposal",
            }
        )

    normalized_items.sort(
        key=lambda item: (
            order_map.get(item["label"], 999),
            int(item["span_start"]),
            item["node_id"],
        )
    )

    last_progress_node_id = root.node_id
    for depth_idx, item in enumerate(normalized_items, start=1):
        label = str(item["label"])
        role = _role_for_label(label, str(item["source"]), is_root=False)
        node = AnchorTreeNode(
            node_id=str(item["node_id"]),
            label=label,
            text=str(item["text"]),
            depth=depth_idx if role not in {AnchorTreeRole.DRIFT, AnchorTreeRole.META} else depth_idx,
            role=role,
            source=str(item["source"]),
            span_start=int(item["span_start"]),
            span_end=int(item["span_end"]),
            repr=item.get("repr"),
            score=float(item["score"]),
            drift_flag=role in {AnchorTreeRole.DRIFT, AnchorTreeRole.META},
        )
        nodes.append(node)
        if role in {AnchorTreeRole.DRIFT, AnchorTreeRole.META}:
            parent_id = last_progress_node_id
            relation = AnchorTreeRelation.ALTERNATIVE_TO
        else:
            parent_id = last_progress_node_id
            relation = AnchorTreeRelation.EXPECTED_NEXT if parent_id != root.node_id else AnchorTreeRelation.CHILD
            last_progress_node_id = node.node_id
        edges.append(AnchorTreeEdge(parent_id=parent_id, child_id=node.node_id, relation=relation, score=float(node.score)))

    tree = make_anchor_tree(
        tree_id=f"observed_{resolved_domain or 'unknown'}",
        root=root,
        nodes=nodes,
        edges=edges,
        domain=resolved_domain or "unknown",
        source_kind="observed",
    )
    tree.metadata["input_text"] = text
    tree.metadata["anchor_count"] = len(anchor_payloads)
    tree.metadata["future_hint_count"] = len(future_hint_candidates)
    tree.metadata["auxiliary_proposal_count"] = len(auxiliary_proposals)
    return tree

