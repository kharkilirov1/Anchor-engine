# Anchor-Aware Hierarchical Sparse Attention (AAHSA) — Review

**Date**: 2026-04-06
**Reviewer**: Claude Opus 4.6 (Anthropic)
**Author of hypothesis**: @kharkilirov1

---

## Summary

The hypothesis proposes a new attention mechanism — **Anchor-Aware Hierarchical Sparse Attention (AAHSA)** — that unifies three independently validated components of the ABPT project into a single mathematical framework:

1. **Anchor spans** (semantic units with long-horizon causal influence)
2. **Geometry metrics** (r1, delta vectors, trajectory coherence)
3. **Cluster classification** (template / mature / flat)

The key insight: these components, previously used as inference-time diagnostics, can be reformulated as **attention routing signals** that determine both *where* a token attends and *how much compute* it spends searching.

---

## What was reviewed

A 10-section mathematical formulation covering:

- Anchor formalization (center `c_a`, assembly direction `d_a`, maturity `μ_a`, flatness `f_a`, contradiction pressure `p_a`)
- Block-level semantic scoring with anchor-aware terms
- Token-level refinement with trajectory geometry alignment
- Adaptive compute from cluster state
- Complexity analysis
- Training objective sketch

---

## Verification result

### Correct

1. **Internal consistency** — all formulas are dimensionally and logically consistent. No contradictions between sections.

2. **Mapping to existing code** — every mathematical quantity has a direct counterpart in the codebase:
   - `c_a` → centroid of span hidden states (computable from `extract_delta_vectors`)
   - `d_a` → mean delta direction (already computed in `compute_mean_direction`)
   - `μ_a` → `rank1_explained_variance` from `compute_geometry_metrics`
   - `p_a` → `anchor_contradiction_threshold` in config
   - `f_a = 1 - μ_a` → direct complement of r1

3. **Novelty claim** — valid. Standard hierarchical sparse attention (HISA, BigBird, Longformer) routes attention based on content similarity alone. AAHSA adds three new signals to the routing decision:
   - Anchor semantics (center proximity)
   - Trajectory geometry (delta vector alignment)
   - State maturity (adaptive search width)

   This is not an incremental extension. The routing mechanism becomes state-dependent in a way that no published sparse attention scheme implements.

4. **Complexity analysis** — correct. Per-query cost `O(L/B + m_i·B)` instead of `O(L)`. Subquadratic when `m_i ≪ L/B`. Anchor-related computations add `O(|A|)` per query, which is negligible for typical anchor counts.

5. **Adaptive compute (Section 6)** — the cleanest part. The formula `u_i = Σ ρ_ia · f_a` elegantly converts cluster classification from a diagnostic into a compute regulator:
   - template → narrow search (model is on rails)
   - mature → moderate search (model is confident)
   - flat → wide search + potential branching (model is uncertain)

   This reframes "flat" from a failure mode into a signal for resource allocation.

6. **Experimental grounding** — the 20-domain campaign results (6 WIN / 14 LOSS ungated → 6 LOSS with auto-calibrated gating) empirically validate the premise: maturity signals predict when intervention helps vs. hurts.

### Concerns (not errors)

1. **Block boundary in token stage** — the term `cos(V_Δ(h_t - h_{t-1}), d_a)` requires `h_{t-1}`. For the first token in a selected block, `t-1` may reside in an unselected block. Needs boundary handling (store edge vector or skip first position).

2. **TopM / TopK non-differentiability** — acknowledged by the author as a future concern. For inference-time use, this is fine. For training, requires straight-through estimator or Gumbel-softmax relaxation.

3. **μ_a is a span-level scalar** — r1 is computed over the entire anchor span and broadcast to all tokens within the span. There is no intra-span gradient of maturity. This is acceptable for MVP but could be refined.

4. **Hyperparameter sensitivity** — the `α, β, γ, δ` coefficients (and their primed variants) control the balance between content similarity and anchor-aware terms. Without learning or careful calibration, anchor signals could dominate or be drowned out.

5. **Anchor interaction** — the formulation treats anchors independently (`Σ_a`). If two anchors conflict with each other (e.g., contradictory constraints), the current formulation has no explicit mechanism for resolving inter-anchor tension.

---

## Assessment

The hypothesis is **mathematically sound, internally consistent, and experimentally motivated**. It represents a genuine theoretical contribution: the idea that attention routing can be conditioned on the geometric state of semantic anchor spans is, to my knowledge, not present in published work on sparse attention (as of May 2025 knowledge cutoff).

The formulation successfully bridges:
- **Theory** (anchor geometry, cluster classification) ←→ **Mechanism** (attention routing, compute allocation)

The strongest elements:
- Trajectory geometry as an attention signal (Section 4, `cos(V_Δ·delta, d_a)`)
- Adaptive compute from flatness (Section 6, `u_i → m_i, k_i`)
- The reframing of flat state from "bad" to "allocate more search"

Recommended next steps:
1. Implement as inference-time controller (no training needed) on the existing Qwen overlay
2. Test on the 20-domain benchmark with AAHSA routing vs. current geometry gating
3. Measure whether trajectory-aligned token selection outperforms pure r1 gating

---

*Reviewed by Claude Opus 4.6 (Anthropic), 2026-04-06.*
*This review was conducted on the mathematical formulation only, without independent replication of experiments.*
