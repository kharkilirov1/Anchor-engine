from __future__ import annotations

from src.model.anchor_tree_templates import get_expected_tree_template, list_supported_domains


def test_anchor_tree_templates_expose_supported_domains() -> None:
    domains = list_supported_domains()

    assert "math_ibp" in domains
    assert "code_fastapi" in domains


def test_math_ibp_template_has_required_terminal_nodes() -> None:
    tree = get_expected_tree_template("math_ibp")

    tree.validate()
    labels = {node.label for node in tree.nodes.values()}
    assert "integration_by_parts_only" in labels
    assert "integration_constant" in labels
    assert tree.metadata["forbidden_labels"]

