from dataclasses import dataclass


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
    branch_diversity_target: float = 0.08

    # Verifier
    use_verifier: bool = True
    verifier_entropy_weight: float = 0.4
    verifier_agreement_weight: float = 0.4
    verifier_consistency_weight: float = 0.2
    verifier_temperature: float = 4.0

    # Plastic Layer
    use_plastic: bool = True
    plastic_lr: float = 1e-4
    plastic_decay: float = 0.99
    plastic_l2_weight: float = 0.01
    plastic_hidden: int = 64
    plastic_noise_scale: float = 0.05
    plastic_mask_ratio: float = 0.15
    plastic_train_updates: int = 1

    # Equilibrium / Routing (Phase 0)
    eq_momentum: float = 0.1
    eq_warmup_steps: int = 50
    router_lr: float = 3e-5
    router_warmup_steps: int = 500
    router_entropy_weight: float = 0.01
    route_temperature: float = 8.0
    route_threshold_momentum: float = 0.2
    route_threshold_offset_scale: float = 0.2
    route_forward_target: float = 0.55
    route_branch_target: float = 0.25
    route_backward_target: float = 0.15
    route_plastic_target: float = 0.05

    # Anchor V1
    use_fog_flow: bool = False
    fog_task_profile: str = "auto"
    fog_compare_ratio: float = 0.25
    fog_memory_ratio: float = 0.75
    fog_expand_ratio: float = 2.0
    fog_gate_ratio: float = 0.125
    anchor_prior_weight: float = 1.0
    anchor_runtime_weight: float = 1.0
    anchor_threshold: float = 0.65
    anchor_domain_mode: str = "auto"
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
    anchor_context_min_viability: float = 0.30
    anchor_use_future_proposal_head: bool = True
    anchor_future_proposal_trigger: float = 0.35
    anchor_future_proposal_hidden: int = 64
    anchor_future_proposal_threshold: float = 0.58
    anchor_future_proposal_temperature: float = 0.75
    anchor_future_proposal_horizon_scale: float = 4.0
    anchor_future_proposal_span_scale: float = 4.0
    anchor_future_proposal_max_horizon: int = 32
    anchor_future_proposal_max_windows: int = 48
    anchor_future_proposal_topk: int = 4
    anchor_future_proposal_residual_scale: float = 0.10
    anchor_proposal_score_weight: float = 0.05
    anchor_proposal_margin_weight: float = 0.05
    anchor_proposal_alignment_weight: float = 0.02
    anchor_proposal_counterfactual_weight: float = 0.05
    anchor_proposal_margin_target: float = 0.05
    anchor_proposal_target_temperature: float = 0.15
    anchor_proposal_counterfactual_margin: float = 0.02
    anchor_proposal_counterfactual_window: int = 4
    anchor_use_proposal_rollout: bool = True
    anchor_proposal_rollout_steps: int = 4
    anchor_proposal_rollout_hidden: int = 64
    anchor_proposal_rollout_weight: float = 0.05
    anchor_proposal_rollout_margin: float = 0.02
    anchor_proposal_rollout_residual_scale: float = 0.15
    anchor_proposal_rollout_pressure_trigger: float = 0.45
    anchor_proposal_rollout_score_trigger: float = 0.90

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

SCALEUP_CONFIG = ModelConfig(
    vocab_size=512, d_model=512, n_heads=8,
    n_layers=4, d_ff=1024, max_seq_len=128,
    plastic_hidden=128,
    anchor_threshold=0.2, anchor_ttl_init=4.0,
    anchor_dead_end_threshold=0.5
)

PRESETS = {
    "baseline-0": BASELINE_0,
    "baseline-1-attnres": BASELINE_1_ATTNRES,
    "baseline-2-branches": BASELINE_2_BRANCHES,
    "baseline-3-plastic": BASELINE_3_PLASTIC,
    "full": FULL_MODEL,
    "toy": TOY_CONFIG,
    "scaleup": SCALEUP_CONFIG,
}
