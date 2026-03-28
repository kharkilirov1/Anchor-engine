# Anchor Span Framework — Research Memo

Date: 2026-03-26
Status: active theory memo, supersedes ABPT-as-module-stack as the main conceptual center

## 1. Why this memo exists

This note captures the current theoretical core so it does not get diluted across chat context, memory, or intermediate code states.

Current conclusion:
- the project center is no longer "a stack of AttnRes + branches + verifier + plasticity"
- the project center is an anchor-centric theory of generation, hallucination, and inference-time self-learning
- existing ABPT modules should now be treated as candidate mechanisms that may serve this theory, not as the theory itself

AttnRes remains useful, but it is a borrowed anti-dilution mechanism from the Kimi/Moonshot line, not the original conceptual core of this project.

## 2. Core intuition

Not all parts of context are equal.
Most tokens are locally useful but do not strongly constrain distant continuation.
Some context-emergent units radically reshape the space of plausible futures.
These units are anchors.

Important update:
- the right unit is not an anchor token
- the right unit is an anchor span

Examples:
- for all
- REST API
- rate limiting
- allergy to penicillin
- proof by contradiction
- under hard real-time constraints

Anchorhood is not reducible to the sum of token importances.
It emerges compositionally in context.

## 3. Main objects

### 3.1 Anchor span
A contiguous span x_{i:j} whose presence substantially changes the future trajectory of generation.

Working definition:
An anchor span is a context-emergent semantic unit with long-horizon causal influence on plausible continuations.

### 3.2 Semantic weight
Omega(x_{i:j}) measures how strongly the span changes the future continuation space.

Interpretation:
- low Omega: noise / local filler
- medium Omega: ordinary useful context
- high Omega: anchor span
- very high, long-lived Omega: super-anchor span

### 3.3 Anchor tree
An anchor is not only a weighted unit; it is the root of a future dependency tree.
The tree consists of downstream spans, constraints, modes of reasoning, or allowable moves that become conditioned by the anchor.

Examples:
- `for all n` induces quantifier commitments and proof obligations
- `proof by contradiction` induces a different proof regime
- `allergy to penicillin` constrains future medical actions

A strong anchor can therefore be described not only by immediate strength but also by the structure of the tree it generates.

## 4. Anchor score and runtime detection

### 4.1 Prior + runtime combination
A practical anchor detector should combine:
- prior anchor knowledge over spans / n-grams / semantic templates
- runtime contextual evidence from hidden-state transitions

Working formula:

A(a) = sigma(w1 * S_prior(a) + w2 * S_runtime(a))

Where:
- a is a candidate span
- A(a) is the current anchor score
- S_prior(a) is prior importance learned over spans/templates
- S_runtime(a) is online contextual evidence that this span reorganized the future trajectory

### 4.2 Runtime signal
Because hidden states are contextualized, runtime scoring can detect the completion or emergence of a span-level anchor even when the input stream is tokenized.

Examples of proxies:
- D(h_t, h_{t-1})
- D(h_t, phi(h_{t-w:t-1}))

This should be interpreted as online evidence that a context-emergent unit has just closed or sharply changed semantic mode.

## 5. Hallucination as anchor failure

The project now treats many hallucinations not as generic uncertainty alone, but as anchor failures.

Three primary failure modes:
- Anchor miss — the model failed to notice a true anchor
- Anchor misread — the model noticed the anchor but interpreted it incorrectly
- Anchor decay — the model initially captured the anchor but failed to preserve it for its semantic lifetime

This yields a first-order hallucination law:

P_hall = 1 - (1 - r)^K

Where:
- K = number of critical anchors required by the task
- r = error probability per anchor

Interpretation:
- narrative tasks often require fewer critical anchors
- code requires more
- mathematics is especially brittle because anchor count and dependency depth are high

## 6. False anchors and dead-end trees

A false anchor may still be locally strong and plausible.
Its danger is that it can generate a false dependency tree.
At first the tree can look coherent, but contradiction pressure accumulates downstream.

### 6.1 Survival model
Let T_d(a) be the dependency tree of anchor a up to depth d.
Let h_v(a) be the contradiction hazard at node v.
Then a survival-style bound is:

S_d(a) = Product_{v in T_d(a)} (1 - h_v(a))

If a nontrivial fraction of descendants carries hazard at least q > 0, then:

P_dead(a,d) >= 1 - (1 - q)^(rho * N_d)

Where:
- rho = fraction of risky descendants
- N_d = number of descendants up to depth d

If N_d grows with depth, dead-end probability rises rapidly.

### 6.2 Tree-mass approximation
A more compact approximation:

P_dead(a,d) ~= 1 - exp(-lambda * M(a,d))

Where:
- M(a,d) is the weighted mass of the anchor tree up to depth d
- lambda > 0 controls how quickly contradiction accumulates

Interpretation:
- true anchors generate viable trees
- false anchors can generate self-undermining trees
- strong false anchors are especially dangerous because they generate large trees before collapsing

## 7. Dead-end recognition and self-correction

A key research claim is that intelligence may depend not only on preserving true anchors, but on recognizing when a chosen anchor is leading into a dead end.

This suggests a staged anchor lifecycle:
- candidate anchor
- provisional anchor
- confirmed anchor
- decaying anchor
- false / dead-end anchor

A central rule:
An anchor should not be consolidated merely because it appeared; it should earn consolidation by surviving future consistency pressure.

### 7.1 Self-learning hypothesis
Anchor spans may be the correct units of online credit assignment during inference.

That is:
- the model should not adapt uniformly over all tokens
- it should adapt around anchors, because anchors organize long-horizon continuation

This reframes plasticity:
- not generic fast adaptation
- but anchor-conditioned inference-time learning

### 7.2 Learning events
Potential high-value update triggers:
- anchor novelty
- anchor conflict
- anchor misread evidence
- anchor decay evidence
- false-anchor collapse

False anchors are especially informative because they reveal where the model committed to the wrong semantic root.

## 8. Relation to existing ABPT modules

### 8.1 Core theory vs mechanisms
Core theory:
- anchor spans
- anchor trees
- anchor miss / misread / decay
- false-anchor collapse
- anchor-based self-learning

Candidate mechanisms:
- AttnRes: borrowed anti-dilution module, may help preserve anchors through depth
- Branches: potentially reinterpretable as competing anchor readings
- Verifier: potentially reinterpretable as anchor-reading selector / contradiction monitor
- Plasticity: potentially reinterpretable as anchor-conditioned online adaptation
- Stage-B routing: currently only a rough scaffold, likely needs redesign around anchor events rather than generic ED alone

### 8.2 Current repo status in light of the theory
- Stage A code implements a module stack, not the full anchor theory
- AttnRes is the closest faithful mechanism currently present
- branches/verifier/plasticity are only partial matches to the new theory
- Stage B router is not yet aligned with anchor-span / anchor-tree concepts
- current evaluation mostly tests trainability, not the anchor theory directly

## 9. Predictions

If the framework is right, we should observe:
- tasks with more critical anchors should fail superlinearly or near-exponentially under anchor errors
- mathematics should be especially sensitive to anchor miss/misread/decay
- false anchors should produce locally plausible but globally brittle continuation trees
- there should exist a dead-end recognition point before final collapse
- anchor-aware revision should outperform generic uncertainty-based correction in anchor-heavy tasks

## 10. Immediate research priorities

1. Formalize anchor tree quantities:
   - depth
   - breadth
   - mass
   - viability
   - contradiction pressure

2. Build toy mathematical cases with:
   - true anchor
   - false anchor
   - explicit descendant tree
   - first contradiction point
   - dead-end recognition point

3. Re-evaluate existing modules only through the new theory:
   - what preserves anchors?
   - what tests anchor interpretations?
   - what revises false anchors?

4. Delay major code redesign until the anchor framework is more stable.

## 11. Honest status

This framework is not yet an empirical proof about real LLM internals.
Current status:
- conceptually unified
- mathematically motivated
- internally consistent at first order
- empirically unverified

That is the correct status to preserve.
