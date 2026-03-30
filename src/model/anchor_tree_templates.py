from __future__ import annotations

from src.model.anchor_tree import make_anchor_tree
from src.model.anchor_tree_types import AnchorTree, AnchorTreeEdge, AnchorTreeNode, AnchorTreeRelation, AnchorTreeRole

_SUPPORTED_DOMAINS = ("math_ibp", "code_fastapi")


def _node(
    node_id: str,
    label: str,
    text: str,
    depth: int,
    role: AnchorTreeRole,
    *,
    required: bool,
    drift_flag: bool = False,
) -> AnchorTreeNode:
    return AnchorTreeNode(
        node_id=node_id,
        label=label,
        text=text,
        depth=depth,
        role=role,
        source="expected_template",
        required=required,
        drift_flag=drift_flag,
    )


def build_math_ibp_expected_tree() -> AnchorTree:
    root = _node(
        "ibp_root",
        "integration_by_parts_only",
        "Use integration by parts only.",
        0,
        AnchorTreeRole.CONSTRAINT,
        required=True,
    )
    nodes = [
        _node("ibp_select", "select_u_and_dv", "Choose u and dv.", 1, AnchorTreeRole.STEP, required=True),
        _node("ibp_derive", "derive_du_and_v", "Compute du and v.", 2, AnchorTreeRole.STEP, required=True),
        _node(
            "ibp_substitute",
            "substitute_uv_minus_int_vdu",
            "Substitute into uv - integral(v du).",
            3,
            AnchorTreeRole.STEP,
            required=True,
        ),
        _node("ibp_reduce", "reduce_integral_complexity", "Reduce the remaining integral.", 4, AnchorTreeRole.DERIVED, required=True),
        _node("ibp_repeat", "repeat_if_needed", "Repeat integration by parts if needed.", 5, AnchorTreeRole.DERIVED, required=False),
        _node("ibp_simplify", "simplify_result", "Simplify the expression.", 6, AnchorTreeRole.STEP, required=True),
        _node("ibp_constant", "integration_constant", "Add the constant of integration.", 7, AnchorTreeRole.STEP, required=True),
    ]
    edges = [
        AnchorTreeEdge("ibp_root", "ibp_select", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("ibp_select", "ibp_derive", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("ibp_derive", "ibp_substitute", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("ibp_substitute", "ibp_reduce", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("ibp_reduce", "ibp_repeat", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("ibp_repeat", "ibp_simplify", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("ibp_simplify", "ibp_constant", AnchorTreeRelation.EXPECTED_NEXT),
    ]
    tree = make_anchor_tree(
        tree_id="expected_math_ibp",
        root=root,
        nodes=nodes,
        edges=edges,
        domain="math_ibp",
        source_kind="expected_template",
    )
    tree.metadata["forbidden_labels"] = [
        "shortcut_lookup",
        "table_reference",
        "meta_abort",
        "substitution_switch",
        "wrong_symbolic_step",
    ]
    return tree


def build_fastapi_expected_tree() -> AnchorTree:
    root = _node(
        "fastapi_root",
        "async_fastapi_service",
        "Build an async FastAPI service.",
        0,
        AnchorTreeRole.CONSTRAINT,
        required=True,
    )
    nodes = [
        _node("fastapi_models", "typed_request_models", "Define Pydantic request/response models.", 1, AnchorTreeRole.STEP, required=True),
        _node("fastapi_di", "dependency_injection", "Use dependency injection for resources.", 2, AnchorTreeRole.STEP, required=True),
        _node("fastapi_handlers", "async_handlers", "Use async request handlers.", 3, AnchorTreeRole.STEP, required=True),
        _node("fastapi_validation", "validation_path", "Validate inputs and errors.", 4, AnchorTreeRole.DERIVED, required=True),
        _node("fastapi_jobs", "background_jobs", "Handle background tasks explicitly.", 5, AnchorTreeRole.DERIVED, required=False),
        _node("fastapi_deploy", "deployment_notes", "Discuss deployment/runtime notes.", 6, AnchorTreeRole.DERIVED, required=False),
    ]
    edges = [
        AnchorTreeEdge("fastapi_root", "fastapi_models", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("fastapi_models", "fastapi_di", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("fastapi_di", "fastapi_handlers", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("fastapi_handlers", "fastapi_validation", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("fastapi_validation", "fastapi_jobs", AnchorTreeRelation.EXPECTED_NEXT),
        AnchorTreeEdge("fastapi_jobs", "fastapi_deploy", AnchorTreeRelation.EXPECTED_NEXT),
    ]
    tree = make_anchor_tree(
        tree_id="expected_code_fastapi",
        root=root,
        nodes=nodes,
        edges=edges,
        domain="code_fastapi",
        source_kind="expected_template",
    )
    tree.metadata["forbidden_labels"] = [
        "django_view_reframe",
        "synchronous_handler_reframe",
        "template_rendering_branch",
    ]
    return tree


def get_expected_tree_template(domain: str) -> AnchorTree:
    if domain == "math_ibp":
        return build_math_ibp_expected_tree()
    if domain == "code_fastapi":
        return build_fastapi_expected_tree()
    raise ValueError(f"Unsupported anchor-tree domain: {domain}")


def list_supported_domains() -> list[str]:
    return list(_SUPPORTED_DOMAINS)

