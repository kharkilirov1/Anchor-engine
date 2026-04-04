from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Backbone
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1

    # Attention Residuals
    use_attn_res: bool = True

    # Branching
    use_branches: bool = True
    n_branches: int = 2
    diversity_weight: float = 0.1

    # Verifier
    use_verifier: bool = True
    verifier_entropy_weight: float = 0.4
    verifier_agreement_weight: float = 0.4
    verifier_consistency_weight: float = 0.2

    # Plastic Layer
    use_plastic: bool = True
    plastic_lr: float = 1e-4
    plastic_decay: float = 0.99
    plastic_l2_weight: float = 0.01
    plastic_hidden: int = 64

    # Equilibrium / Routing (Phase 0)
    eq_momentum: float = 0.1
    eq_warmup_steps: int = 50
    router_lr: float = 3e-5
    router_warmup_steps: int = 500
    router_entropy_weight: float = 0.01

    # Anchor V1
    anchor_prior_weight: float = 1.0
    anchor_runtime_weight: float = 1.0
    anchor_threshold: float = 0.65
    anchor_max_candidates: int = 6
    anchor_ttl_init: float = 4.0
    anchor_support_decay: float = 0.9
    anchor_candidate_promote_threshold: float = 0.55
    anchor_confirm_threshold: float = 0.7
    anchor_revision_threshold: float = 0.35
    anchor_contradiction_threshold: float = 0.65
    anchor_dead_end_threshold: float = 0.85
    anchor_arbiter_beta: float = 8.0
    anchor_arbiter_revise_threshold: float = 0.45
    anchor_revision_temperature: float = 1.0
    anchor_viability_alpha: float = 2.0
    anchor_viability_beta: float = 2.5
    anchor_age_gamma: float = 0.25
    anchor_descendant_mass_delta: float = 0.75
    anchor_descendant_coherence_eta: float = 0.75
    anchor_detector_alignment_weight: float = 0.05
    anchor_context_stability_weight: float = 0.01
    anchor_dependency_threshold: float = 0.55
    anchor_dependency_confirm_slope: float = 0.10
    anchor_dependency_temporal_window: float = 16.0
    anchor_dependency_similarity_weight: float = 0.55
    anchor_dependency_temporal_weight: float = 0.20
    anchor_dependency_support_weight: float = 0.15
    anchor_dependency_viability_weight: float = 0.10
    anchor_dependency_max_predecessors: int = 4
    anchor_dependency_counterfactual_top_edges: int = 0
    anchor_dependency_future_window: int = 16

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    batch_size: int = 32
    eval_interval: int = 100
    gradient_clip: float = 1.0


# Ablation presets
BASELINE_0 = ModelConfig(
    use_attn_res=False, use_branches=False,
    use_verifier=False, use_plastic=False,
)

BASELINE_1_ATTNRES = ModelConfig(
    use_attn_res=True, use_branches=False,
    use_verifier=False, use_plastic=False,
)

BASELINE_2_BRANCHES = ModelConfig(
    use_attn_res=True, use_branches=True,
    use_verifier=True, use_plastic=False,
)

BASELINE_3_PLASTIC = ModelConfig(
    use_attn_res=True, use_branches=False,
    use_verifier=False, use_plastic=True,
)

FULL_MODEL = ModelConfig(
    use_attn_res=True, use_branches=True,
    use_verifier=True, use_plastic=True,
)

TOY_CONFIG = ModelConfig(
    vocab_size=512, d_model=64, n_heads=2,
    n_layers=3, d_ff=128, max_seq_len=128,
    plastic_hidden=16,
)

CONFIG_500M = ModelConfig(
    vocab_size=32000, d_model=1280, n_heads=20,
    n_layers=24, d_ff=5120, max_seq_len=256,
    batch_size=1, plastic_hidden=256,
    use_attn_res=True, use_branches=True,
    use_verifier=True, use_plastic=True,
)

CONFIG_150M = ModelConfig(
    vocab_size=32000, d_model=768, n_heads=12,
    n_layers=12, d_ff=3072, max_seq_len=256,
    batch_size=2, plastic_hidden=128,
    use_attn_res=True, use_branches=True,
    use_verifier=True, use_plastic=True,
)

CONFIG_150M_BASELINE = ModelConfig(
    vocab_size=32000, d_model=768, n_heads=12,
    n_layers=12, d_ff=3072, max_seq_len=256,
    batch_size=2, plastic_hidden=128,
    use_attn_res=False, use_branches=False,
    use_verifier=False, use_plastic=False,
)

CONFIG_150M_V16 = ModelConfig(
    vocab_size=16000, d_model=768, n_heads=12,
    n_layers=12, d_ff=3072, max_seq_len=256,
    batch_size=2, plastic_hidden=128,
    use_attn_res=True, use_branches=True,
    use_verifier=True, use_plastic=True,
)

CONFIG_150M_V16_BASELINE = ModelConfig(
    vocab_size=16000, d_model=768, n_heads=12,
    n_layers=12, d_ff=3072, max_seq_len=256,
    batch_size=2, plastic_hidden=128,
    use_attn_res=False, use_branches=False,
    use_verifier=False, use_plastic=False,
)

PRESETS = {
    "baseline-0": BASELINE_0,
    "baseline-1-attnres": BASELINE_1_ATTNRES,
    "baseline-2-branches": BASELINE_2_BRANCHES,
    "baseline-3-plastic": BASELINE_3_PLASTIC,
    "full": FULL_MODEL,
    "toy": TOY_CONFIG,
    "500m": CONFIG_500M,
    "150m": CONFIG_150M,
    "150m-baseline": CONFIG_150M_BASELINE,
    "150m-v16": CONFIG_150M_V16,
    "150m-v16-baseline": CONFIG_150M_V16_BASELINE,
}
