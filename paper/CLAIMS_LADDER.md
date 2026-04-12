# Claims Ladder for FOG Architecture

## Level A — Strongly Supported by Current Data
These claims are backed by quantitative, multi-seed, or causal intervention data.
- **Task-Selective Specialization:** FOG architecture exhibits stronger and more isolated causal dissociation of functional pathways (attention heads) compared to uniform baselines, specifically separating localized matching tasks from broad retrieval tasks (Exp9).
- **Capacity Starvation Resilience:** FOG sustains computational performance at significantly lower parameter counts (down to ~7K parameters, d_model=16) than uniform architectures on memory-intensive tasks (Exp4).
- **Reduced Catastrophic Interference:** When constrained by tight parameter budgets (~250K-450K), FOG mitigates destructive interference between conflicting task objectives (e.g., sharp focus vs. broad context) better than monolithic architectures (Exp8 Multi-task).
- **Subspace Preservation:** FOG maintains higher effective rank (eRank) in its memory projections during training, whereas uniform models exhibit stronger dimensional collapse when forced to multiplex tasks (Exp1/3).

## Level B — Supported but Still Interpretive
These claims are consistent with empirical signatures but rely on theoretical mapping.
- **Morphology Drives Pathway Stratification:** Heterogeneous constraints (e.g., narrow compare, wide memory) encourage cleaner separation of functional pathways.
- **Task-Dependent Signature Profiles:** The mathematical signatures of FOG layers (attention entropy, gate polarization) naturally align with theoretical cognitive motifs (e.g., sharp attention for "compare/read", high gate variance for "select/route") (Exp5).
- **Cross-Layer Drift Isolation:** FOG's gating mechanism allows it to protect representations in memory pathways, resulting in significantly lower cross-layer angular drift compared to the entangled drift seen in uniform models (Exp2).

## Level C — Not Yet Established (Future Work)
These are theoretical extrapolations that require further validation.
- FOG universally improves long-horizon mathematical or logical reasoning in large language models.
- FOG directly translates to superior performance on unstructured, real-world terminal operation or agentic tasks.
- FOG maintains its relative efficiency advantage at the frontier scale (10B+ parameters).
