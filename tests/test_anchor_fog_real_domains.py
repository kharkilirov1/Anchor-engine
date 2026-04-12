from scripts.run_anchor_fog_real_domains import (
    build_capacity_matches,
    build_domain_specs,
    build_model_specs,
)


def test_real_domain_specs_cover_only_real_domains() -> None:
    specs = build_domain_specs(steps=100)
    assert [spec.key for spec in specs] == ["stories", "code", "math"]
    assert all(spec.dataset != "anchor-synthetic" for spec in specs)


def test_real_domain_model_specs_include_anchor_fog() -> None:
    specs = build_model_specs()
    keys = [spec.key for spec in specs]
    assert keys == ["baseline", "anchor", "anchor_fog"]
    assert specs[-1].use_fog_flow is True


def test_equal_param_matching_keeps_models_close() -> None:
    specs = build_domain_specs(steps=10)
    matches, default_counts, target_params = build_capacity_matches(specs[0])
    assert set(matches) == set(default_counts) == {"baseline", "anchor", "anchor_fog"}
    deltas = [abs(match.param_count - target_params) for match in matches.values()]
    assert max(deltas) <= max(50_000, int(target_params * 0.02))
