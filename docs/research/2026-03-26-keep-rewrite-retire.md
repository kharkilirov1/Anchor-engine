# Keep, Reinterpret, Rewrite, Retire — Current ABPT Audit

Date: 2026-03-26
Status: active transition memo
Related:
- `docs/research/2026-03-26-anchor-framework.md`
- `docs/research/2026-03-26-anchor-objects.md`
- `docs/research/2026-03-26-architecture-requirements.md`
- `docs/research/PROJECT_MAP.md`

## 1. Purpose

This note audits the current ABPT codebase against the anchor-centric architecture requirements.
The goal is to decide what should be:
- kept as-is
- kept but reinterpreted
- rewritten
- retired as a conceptual center

This is not yet a rewrite plan.
It is a strategic mapping from the old codebase to the new theory.

## 2. Decision categories

### Keep
The component already serves the new theory reasonably well and can remain a direct building block.

### Reinterpret
The component is still valuable, but only under a new role in the anchor framework.

### Rewrite
The current implementation or abstraction is too tied to the old framing and should not be carried forward in its present form.

### Retire as center
The component or idea may remain historically useful, but it should no longer define the project's main identity.

## 3. Summary table

| Component | Decision | Short reason |
|---|---|---|
| Transformer backbone | Keep | still the right general sequence substrate |
| AttnRes | Keep + reinterpret | useful anti-dilution mechanism, but not the project core |
| Stage A integrated model (`abpt.py`) | Keep as historical scaffold | good modular baseline, not final architecture |
| Branches | Reinterpret + rewrite | concept survives, current implementation too shallow |
| Verifier | Reinterpret + rewrite | role survives, current scoring too weak for anchor arbitration |
| Plastic layer | Reinterpret + rewrite | idea survives, current adapter is not anchor-conditioned learning |
| Stage B routing (`abpt_b.py`) | Rewrite | current ED-centric scaffold is misaligned with anchor theory |
| Equilibrium signal | Retire as center / maybe keep as proxy | may remain a useful signal, but should not define routing logic |
| Scatter/gather / token buckets | Keep only as systems idea | efficiency trick may survive, but not as theoretical center |
| Current ablation framing | Keep as historical evidence | useful archive, no longer enough as the main research program |

## 4. Component-by-component audit

## 4.1 Transformer backbone
Files:
- `src/model/backbone.py`
- parts of `src/model/abpt.py`
- parts of `src/model/abpt_b.py`

### Decision
Keep.

### Why
The new theory still needs a general sequence-processing substrate.
Nothing in the anchor framework requires abandoning the transformer backbone outright.

### Caveat
The backbone is no longer the whole story.
It becomes the base medium through which anchor-specific structures operate.

### Best interpretation now
A general contextual field over which anchor events, anchor memory, and anchor revision operate.

## 4.2 Attention Residuals (AttnRes)
Files:
- `src/model/attention.py`
- `src/model/backbone.py`

### Decision
Keep, but reinterpret.

### Why
AttnRes is the cleanest currently implemented mechanism that genuinely addresses dilution across depth.
That still matters if anchor spans must survive long enough to influence future generation.

### Why not keep it as the project center
AttnRes is borrowed from an external paper line and only solves one part of the problem: depth preservation.
It does not detect anchors, estimate viability, track descendants, or revise false anchors.

### Best interpretation now
Optional anti-dilution support for anchor preservation.

## 4.3 Stage A integrated model
File:
- `src/model/abpt.py`

### Decision
Keep as historical scaffold, not as final architecture.

### Why
Stage A remains useful as:
- a modular baseline
- an ablation harness
- a sanity-check implementation of the old stack

### Limitation
It assembles modules sequentially but has no explicit notion of:
- anchor span
- anchor memory
- contradiction pressure
- viability
- dead-end revision

### Best interpretation now
Stable historical baseline and sandbox, not the target architecture.

## 4.4 Branches
File:
- `src/model/branches.py`

### Decision
Reinterpret conceptually, rewrite implementation.

### Why the concept survives
Branches still make sense if they become:
- competing anchor readings
- provisional semantic hypotheses around ambiguous anchors

### Why the implementation should be rewritten
Current branches differ mostly by temperature and linear projection.
That is too weak for the new theory.
It does not create meaningfully different anchor interpretations or descendant regimes.

### Best future role
Anchor-aware branching that activates around ambiguous roots or revision events.

## 4.5 Verifier
File:
- `src/model/verifier.py`

### Decision
Reinterpret conceptually, rewrite implementation.

### Why the concept survives
Verifier may become one of the most important modules in the new theory:
- evaluate competing anchor readings
- monitor branch resilience under contradiction pressure
- help identify dead-end recognition points

### Why the implementation is insufficient
Current verifier mainly blends entropy and agreement.
It does not yet reason over:
- descendant obligations
- contradiction accumulation
- anchor viability
- revision timing

### Best future role
Anchor arbiter and contradiction-sensitive branch selector.

## 4.6 Plastic layer
File:
- `src/model/plastic.py`

### Decision
Reinterpret conceptually, rewrite implementation.

### Why the concept survives
Inference-time learning remains central, but it now needs a disciplined object: anchors.
Plasticity still matters if it becomes localized around anchor events.

### Why the implementation is insufficient
Current plastic layer is basically an adapter with decay and an adaptation hook.
It is not yet:
- anchor-triggered
- anchor-localized
- revision-aware
- protected against false-anchor consolidation

### Best future role
Anchor-conditioned self-learning, likely bounded and reversible.

## 4.7 Stage B unified router
File:
- `src/model/abpt_b.py`

### Decision
Rewrite.

### Why
Current Stage B is built around equilibrium deviation and route buckets.
That may have been a useful prototype, but it is not aligned with the new theoretical center.

### What is missing
- no explicit anchor detector
- no anchor memory
- no descendant linkage
- no contradiction pressure model
- no viability estimation
- no dead-end recognition mechanism
- no explicit false-anchor revision logic

### Best interpretation now
A useful historical experiment in selective compute, not the basis of the next architecture.

## 4.8 Equilibrium signal
File:
- `src/model/equilibrium.py`

### Decision
Retire as conceptual center; maybe keep as auxiliary proxy.

### Why
Deviation from running mean is too indirect to serve as the foundation of anchor theory.
It may still correlate with some semantic events, but it does not define the right objects.

### Best future role
At most an auxiliary signal among others, not the master routing principle.

## 4.9 Scatter/gather and route buckets
File:
- `src/model/adaptive_routing.py`

### Decision
Keep only as systems idea, not as theory.

### Why
Grouping tokens or states for efficient compute may still be useful later.
But it belongs to optimization / execution strategy, not to the conceptual heart of the model.

### Best future role
Implementation-level acceleration once anchor-conditioned computation exists.

## 4.10 Current train/eval framing
Files:
- `train.py`
- existing test suite
- archived notebooks and ablations

### Decision
Keep as infrastructure, rewrite the research framing.

### Why
The existing training and test infrastructure proves the repo can run, train, and ablate.
That is still valuable.
But the active theory now requires new evaluation objects:
- true vs false anchors
- contradiction pressure
- dead-end recognition lead time
- anchor revision success

### Best interpretation now
Engineering harness, not sufficient scientific harness.

## 5. What should survive into the next architecture draft

Most likely survivors:
1. transformer backbone
2. AttnRes as optional anti-dilution mechanism
3. the existence of branches, verifier, and plasticity as broad functions
4. parts of the training/test scaffold

Most likely non-survivors in current form:
1. ED-centric Stage B routing as the main theory
2. temperature-only branch specialization
3. entropy/agreement-only verifier
4. generic adapter plasticity without anchor semantics

## 6. What the next design should not do

The next design should not:
- start from old Stage B and merely rename variables to "anchor"
- treat anchor theory as a cosmetic reinterpretation of ED routing
- keep branches, verifier, and plasticity in their current form just because they already exist
- assume that trainability equals theory fidelity

## 7. Clean strategic conclusion

The current repo contains useful parts, but the project center has moved.
The correct strategy is not:
- throw everything away immediately
nor:
- keep the old architecture and force the new theory onto it

The correct strategy is:
1. keep the stable substrate and the few genuinely reusable mechanisms
2. reinterpret broad module roles under the anchor framework
3. rewrite components whose abstractions are anchored to the old theory
4. retire ED-routing and the old module stack as the project identity

## 8. One-sentence operational summary

Keep the backbone, keep AttnRes as a tool, reinterpret branches/verifier/plasticity, and rewrite Stage B around explicit anchors, descendant tracking, contradiction pressure, viability, and revision.
