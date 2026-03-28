# Anchor Span Hypothesis

Status: working hypothesis with preliminary small-scale evidence  
Last updated: 2026-03-28

## 1. Problem framing

Language models hallucinate. A common explanation is that the model simply does not know the answer. This project explores a narrower and more operational hypothesis: the model may sometimes have enough latent knowledge, but fail to preserve the right context spans during generation.

Attention is a finite resource. Softmax distributes a total mass of `1.0` across prior tokens. As context grows, the share available to any single token or phrase tends to shrink. If semantically critical spans lose too much functional influence, generation may drift toward generic but locally plausible continuations.

In this framing, some hallucinations may be understood as failures of **anchor retention**, not only as absence of knowledge.

## 2. Anchor spans

### Definition

An anchor span is a contiguous subsequence of tokens whose removal or replacement substantially changes the distribution of future tokens:

```text
anchor(x_{i:j}) ⟺ KL(P(t>j | context) ‖ P(t>j | context \ x_{i:j})) > δ
```

The key claim is that anchors are often **spans**, not individual tokens. Semantic weight is not necessarily additive:

```text
S("for all") ≠ S("for") + S("all")
S("REST API") ≠ S("REST") + S("API")
S("rate limiting") ≠ S("rate") + S("limiting")
```

Neither component token must be an anchor on its own. The anchor property may emerge from the combination in context.

### Why this matters for implementation

Runtime anchor scoring in this repo operates on hidden states, which are already contextualized. When the model processes `"all"` after `"for"`, the hidden state can already encode something closer to `"for all"` than to either token in isolation. A relative shift such as `‖h_i - h_{i-1}‖ / ‖h_{i-1}‖` can therefore serve as a practical span-sensitive signal without explicit n-gram enumeration.

The prior map, however, is better interpreted as span- or phrase-sensitive rather than as a pure per-token frequency table. A naive per-token prior would over-anchor common words such as `"all"` in contexts where they are not semantically central.

### Rough anchor density by domain

These values are rough working estimates from small manual inspection, not validated measurements:

| Domain | Approximate anchor density `ρ` |
|---|---:|
| Children’s stories | ~0.06 |
| General text | ~0.10 |
| Instructions | ~0.15 |
| Code | ~0.32 |
| Legal text | ~0.40 |
| Mathematical proofs | ~0.85 |

The intended role of these numbers is heuristic ordering, not benchmark truth.

## 3. Formula 1 — Hallucination probability

If a task requires `K` anchors to remain functionally intact, and each has an approximate failure probability `r`, then a first-order estimate for at least one anchor failure is:

```text
P(hallucination) = 1 - (1 - r)^K
```

where `K = ρ × L` in the simplest density-times-length approximation.

### Example ordering (`r = 0.01`)

| Domain | `K` | `P(hallucination)` |
|---|---:|---:|
| Stories | 30 | 26% |
| General text | 50 | 39% |
| Instructions | 75 | 53% |
| Code | 160 | 80% |
| Legal | 200 | 87% |
| Mathematics | 425 | 99% |

This ordering is directionally consistent with the intuition that low-anchor-density domains are easier than high-anchor-density domains. It should be treated as a toy law, not a validated domain-performance model.

### Limitations

The formula assumes independent anchor failures. Real anchors interact: losing one constraint can change the relevance of others. In practice, dependencies may make true failure rates differ substantially from this simple estimate.

## 4. Formula 2 — Combined anchor scoring

Anchor detection currently combines two signals:

```text
S(t_i) = σ(w₁ · S_prior(t_i) + w₂ · S_runtime(t_i))
```

`S_prior` is the learned prior anchor weight.  
`S_runtime` is the online emergence signal from hidden-state change.  
`w₁` and `w₂` are learned combination weights.

In the current implementation, the runtime term is approximated by:

```text
S_runtime(t_i) = ‖h_i - h_{i-1}‖ / ‖h_{i-1}‖
```

This is not a full Bayesian derivation in the strict statistical sense. It is a practical dual-signal scoring rule used by the current prototype.

### Implementation status

- runtime scoring: implemented
- prior map: implemented
- learned signal combination: implemented

What is **not** established yet is whether this scoring rule is the right one beyond small-scale internal experiments.

## 5. Formula 3 — False-anchor collapse

Each anchor induces downstream dependencies: a future structure of tokens whose local correctness depends on the anchor being correct.

If the anchor is false, contradictions may accumulate as descendant mass grows:

```text
P_dead(a, d) ≈ 1 - e^{-λ · M(a, d)}
```

where `M(a, d)` is the effective descendant mass of anchor `a` up to depth `d`, and `λ` is a contradiction-rate parameter.

### Main qualitative prediction

A stronger false anchor may produce a larger descendant structure and therefore reveal itself sooner through accumulated contradiction pressure. In that sense, some strong false anchors may be easier to detect than weak false anchors.

This claim is currently best understood as a branching-process-style intuition supported by toy cases and diagnostics, not as a finished theorem.

### Implementation status

- descendant tracking: implemented
- contradiction pressure: implemented
- viability scoring: implemented
- dead-end counting: implemented

## 6. Current empirical support

### Semantic probe families

Current probe results show stable-vs-conflict separation across several toy families:

| Family | Stable pressure | Conflict pressure | Stable viability | Conflict viability | Conflict dead-ends |
|---|---:|---:|---:|---:|---:|
| quantifier | 0.206 | 0.372 | 0.865 | 0.693 | 1 |
| proof_mode | 0.341 | 0.468 | 0.772 | 0.610 | 1 |
| induction | 0.302 | 0.509 | 0.797 | 0.504 | 2 |
| formal_limit | 0.412 | 0.465 | 0.637 | 0.569 | 1 |

Interpretation: in these toy settings, conflict cases produce higher pressure, lower viability, and more dead-ends than stable cases. That is encouraging, but still narrow evidence.

### TinyStories BPE comparison

Recent small-scale training comparison:

| Metric | Baseline | Anchor | Δ |
|---|---:|---:|---:|
| final `val_bpb` | 6.2377 | 5.9606 | -4.4% |
| best `val_bpb` | 6.2377 | 5.9061 | -5.3% |

Within the anchor run:

| Metric | Start | End | Direction |
|---|---:|---:|---|
| dead-ends | 75 | 53 | improving |
| viability | 0.112 | 0.232 | improving |
| contradiction | 0.843 | 0.782 | improving |
| detector alignment | 0.121 | 0.003 | improving |
| anchors active | 21 | 43 | growing |
| proposal influence | 0.0 | 0.0 | inactive |

This is enough to say the anchor prototype is not obviously inert. It is **not** enough to say that the full anchor theory is validated.

## 7. Anchor failure taxonomy

Three working failure modes:

### Anchor miss
The system fails to identify an anchor at all. It receives no special retention.

### Anchor misclassification
The system selects the wrong anchor span or attaches the wrong properties to it.

### Anchor decay
The system initially identifies the right anchor, but its influence weakens over time despite retention attempts.

The current implementation tries to address all three through detection, monitoring, and memory. Their relative effectiveness is still unresolved.

## 8. Proposal path

When rising contradiction pressure and falling viability suggest that an anchor reading is degrading, the system can generate alternative proposals.

### Implementation status

- proposal detection: implemented
- proposal scoring: implemented
- proposal blending: implemented
- gate calibration: implemented
- alias resolution for descendant anchors: implemented

### Current limitation

The proposal path has **not** yet shown meaningful activation during actual gradient-based training runs. It works on probe-style inputs, but remains at zero influence in current training experiments.

Current working diagnosis:

1. normal training data contains little or no explicit semantic conflict;
2. early in training, the model may not yet have a rich enough internal world model for contradiction-sensitive proposal routing to matter.

This suggests phased training may be more appropriate than training the entire anchor stack from scratch.

## 9. Theoretical extensions not implemented

These ideas are exploratory and are **not** part of the current codebase.

### 9.1 Anchor polarity

Anchors may differ in whether they contract or expand the valid continuation space:

- contracting anchors: impose stronger constraints
- expanding anchors: permit broader continuation sets

One speculative health formulation is:

```text
Health(a, d) = (1-σ(Φ)) · coherence + σ(Φ) · diversity · relevance
where σ(Φ) = (1+Φ)/2
```

Status: theoretical only.

### 9.2 Anchor-gated plasticity

The older plastic memory layer could update primarily on anchor tokens rather than on every token.

Status: theoretical only.

### 9.3 KV-cache replacement

If anchor detection became highly reliable, one long-term possibility would be to store anchor states plus a local window instead of the full KV cache.

Status: long-term research idea only.

### 9.4 Multi-axis verification

Current monitoring mostly checks internal consistency of descendant structure. A stronger system might also compare that structure against the model’s broader learned priors in a more explicit way.

Status: partially implicit in current monitoring; not fully implemented.

## 10. Falsifiable predictions

These are intended as testable consequences, not conclusions:

1. hallucination frequency across domains should correlate with anchor density `ρ`;
2. middle-of-context failure should be stronger for high-`ρ` content than for low-`ρ` content;
3. explicit anchor retention should help more on code/math than on stories;
4. at least part of the vocabulary should show relatively stable anchor prior behavior across contexts;
5. pretrained backbones with an added anchor layer may outperform from-scratch anchor training.

## 11. Relationship to existing work

| Approach | What it does | How this repo differs |
|---|---|---|
| RAG | retrieves external context | this project studies retention of already-present internal context |
| Chain-of-thought | emits explicit reasoning tokens | this project monitors hidden-state dynamics directly |
| Attention sinks | observes persistent attention patterns | this project treats anchor handling as an explicit control problem |
| hierarchical/document methods | use structure or segmentation | this project uses semantic impact rather than explicit document structure |
| reasoning models | spend more tokens on deliberate reasoning | this project aims at hidden-state control with lower token overhead |

## 12. Open questions

- Does anchor detection quality scale with model size?
- Can proposal routing be trained effectively with conflict-supervised curricula?
- Is explicit tree structure necessary, or is pressure/viability monitoring enough?
- Can the anchor layer sit cleanly on top of a pretrained backbone?
- What is the real inference overhead of full anchor monitoring?

## References

- `docs/research/ARCHITECTURE_V1.md`
- `docs/research/PROJECT_MAP.md`
- generated probe/training histories and ad-hoc experiment artifacts are intentionally not versioned in the public repo snapshot
