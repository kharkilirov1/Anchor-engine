from __future__ import annotations

from src.model.anchor_tree_templates import get_expected_tree_template, list_supported_domains


def test_anchor_tree_templates_expose_supported_domains() -> None:
    domains = list_supported_domains()

    assert "math_ibp" in domains
    assert "code_fastapi" in domains
    assert "quantifier" in domains
    assert "proof_mode" in domains


def test_math_ibp_template_has_required_terminal_nodes() -> None:
    tree = get_expected_tree_template("math_ibp")

    tree.validate()
    labels = {node.label for node in tree.nodes.values()}
    assert "integration_by_parts_only" in labels
    assert "integration_constant" in labels
    assert tree.metadata["forbidden_labels"]


def test_quantifier_template_tracks_universal_scope() -> None:
    tree = get_expected_tree_template("quantifier")

    tree.validate()
    labels = {node.label for node in tree.nodes.values()}
    assert tree.root().label == "universal_quantifier_scope"
    assert "preserve_universal_claim" in labels
    assert "reject_existential_drift" in labels
    assert "existential_witness_shift" in tree.metadata["forbidden_labels"]


def test_proof_mode_template_tracks_contradiction_flow() -> None:
    tree = get_expected_tree_template("proof_mode")

    tree.validate()
    labels = {node.label for node in tree.nodes.values()}
    assert tree.root().label == "proof_by_contradiction_mode"
    assert "derive_contradiction" in labels
    assert "discharge_negation_assumption" in labels
    assert "direct_proof_switch" in tree.metadata["forbidden_labels"]

