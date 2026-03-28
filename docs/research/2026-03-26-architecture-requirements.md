# Architecture Requirements from Anchor Theory

Date: 2026-03-26
Status: active design bridge memo
Related:
- `docs/research/2026-03-26-anchor-framework.md`
- `docs/research/2026-03-26-anchor-objects.md`
- `docs/research/2026-03-26-math-toy-cases.md`
- `docs/research/PROJECT_MAP.md`

## 1. Purpose

This note translates the current anchor-centric theory into architecture requirements.
It does not yet specify final implementation.
It defines what any serious next architecture must be able to do if the theory is correct.

The guiding rule is:
- do not design from old modules outward
- design from theory inward

## 2. Core architectural premise

If generation is organized by anchor spans and their descendant structures, then a model needs more than ordinary token-level attention flow.

A sufficient next-generation architecture should support five functions:
1. detect anchors
2. preserve anchors
3. monitor anchor health
4. revise false anchors
5. learn selectively from anchor events

Everything else is secondary.

## 3. Requirement A — Anchor detection

### What the architecture must do
The model must detect when a context-emergent anchor span has formed or sharply changed semantic mode.

### Why
Without anchor detection, the rest of the framework has no object to act on.

### Minimal capability
The model needs an anchor-scoring subsystem that combines:
- prior span knowledge
- runtime contextual evidence

### Desired properties
- span-sensitive, not purely token-local
- robust to context changes
- able to treat the same token differently in different span closures
- cheap enough to run online during generation

### Open implementation options
- hidden-state transition scoring
- span proposal over sliding windows
- learned n-gram / pattern priors
- latent anchor boundary detector

## 4. Requirement B — Anchor memory

### What the architecture must do
The model must maintain anchors beyond their moment of appearance, in proportion to their semantic lifetime.

### Why
Standard transformer flow may not preserve important spans according to how long they matter.
Attention alone is not guaranteed to keep a semantically critical unit alive for the right duration.

### Minimal capability
The model needs an explicit or semi-explicit structure that keeps track of active anchors.

### Anchor memory should minimally store
- anchor identity or representation
- current anchor score
- semantic weight
- state: candidate / provisional / confirmed / decaying / dead-end
- provisional lifetime or decay estimate

### Desired properties
- anchors do not all live equally long
- stronger anchors can persist longer or be refreshed more often
- the system can keep multiple anchors simultaneously

## 5. Requirement C — Descendant tracking

### What the architecture must do
The model must be able to connect future obligations, decisions, or local continuations back to the anchors that generated them.

### Why
Without descendant tracking, the theory cannot distinguish:
- true anchor trees
- false anchor trees
- which future branch belongs to which anchor root

### Minimal capability
The model must maintain at least a coarse dependency relation between active anchors and downstream states.

### Desired properties
- support local descendants and long-horizon descendants
- allow multiple active anchors
- tolerate partial overlap between anchor influences

### First-order note
A true final system may need a DAG rather than a strict tree, but the architecture must at least expose root-descendant linkage.

## 6. Requirement D — Contradiction monitoring

### What the architecture must do
The model must estimate contradiction pressure against active anchors before terminal collapse.

### Why
A system that only notices fully explicit contradiction reacts too late.
The theory requires pre-collapse strain detection.

### Minimal capability
The model needs a contradiction-monitoring signal for each active anchor.

### Potential signal sources
- explicit logical inconsistency
- failure to satisfy expected descendant obligations
- branch divergence around anchor-sensitive steps
- verifier confidence collapse
- rising repair cost / anchor debt
- retrospective superiority of an alternative anchor

### Desired property
Contradiction pressure should rise before final failure, not only at failure.

## 7. Requirement E — Viability estimation

### What the architecture must do
The model must estimate whether an active anchor is still worth defending.

### Why
The key decision is not simply whether an anchor exists, but whether it remains globally livable.

### Minimal capability
The system needs a viability score or latent viability state for each active anchor.

### Architectural role
Viability sits between memory and revision:
- memory keeps anchors alive
- contradiction pressure stresses them
- viability decides whether to preserve, downgrade, or revise them

## 8. Requirement F — Dead-end recognition and revision

### What the architecture must do
The model must identify the earliest point where the current anchor should be revised before full collapse.

### Why
The theory's corrective power depends on acting before the wall, not after impact.

### Minimal capability
The architecture needs a revision trigger that is anchor-specific rather than only generic-uncertainty-based.

### Possible revision actions
- downgrade anchor from confirmed to provisional
- branch into competing anchor readings
- replace current anchor with a better candidate
- retain anchor but lower its control weight
- mark anchor as dead-end and prevent consolidation

### Desired property
There should be a meaningful gap between:
- dead-end recognition point
- final collapse point

A strong architecture widens that gap in the useful direction.

## 9. Requirement G — Branching should become anchor-aware

### What the architecture must do
If branching remains in the architecture, it should be invoked around ambiguous anchor interpretations rather than general uncertainty alone.

### Why
Branches become far more principled when they represent competing anchor readings.

### Minimal capability
The system must be able to say:
- current span may support multiple anchor interpretations
- let these compete temporarily
- let verifier or future evidence decide

### Consequence
This reframes branches from generic hypothesis generators into anchor interpretation workers.

## 10. Requirement H — Verifier should become an anchor arbiter

### What the architecture must do
Verifier should estimate not only branch confidence, but anchor-reading quality and contradiction resilience.

### Why
In the new theory, the key ambiguity is often not "which token next?" but "which anchor interpretation is shaping the future correctly?"

### Minimal capability
Verifier should help evaluate:
- which anchor reading best explains current descendants
- which branch shows lower contradiction pressure
- which reading deserves consolidation

### Consequence
Verifier becomes central to anchor revision, not just branch averaging.

## 11. Requirement I — Plasticity should become anchor-conditioned learning

### What the architecture must do
If inference-time plasticity is kept, it should update around anchor events rather than diffuse token flow.

### Why
The theory suggests anchors are the right units of online credit assignment.

### Minimal capability
Plastic updates should be triggered by events such as:
- anchor novelty
- anchor confirmation
- anchor misread evidence
- anchor decay evidence
- false-anchor collapse

### Desired safety properties
- updates should be local or bounded
- newly learned anchors may need provisional status
- false anchors should not immediately harden into permanent memory
- some updates may need to be reversible or strongly decayed

## 12. Requirement J — Support multiple anchor states

### What the architecture must do
A serious anchor architecture cannot treat anchors as binary present/absent objects.

### Minimal state space
At minimum:
- candidate
- provisional
- confirmed
- decaying
- dead-end

### Why
This keeps the system elegant and brain-like:
- not every hypothesis should be trusted immediately
- not every failing anchor should be dropped instantly
- some anchors should survive only after future support

## 13. Requirement K — Evaluation must change

### What the architecture must do
The evaluation stack must not rely only on aggregate LM loss or bpb.

### Why
The theory makes predictions about structure, not only scalar quality.

### Required new evaluation modes
- true vs false anchor cases
- anchor-heavy vs anchor-light tasks
- dead-end recognition lead time
- false-anchor revision success
- anchor lifetime preservation
- contradiction pressure tracking

### Important consequence
A model can look trainable in language modeling while still failing the anchor theory.

## 14. What likely survives from the current ABPT code

### Likely survivors
- transformer backbone as general sequence substrate
- AttnRes as optional anti-dilution module
- verifier idea, but redefined
- branching idea, but redefined
- plasticity idea, but redefined

### Likely non-survivors in current form
- generic Stage B ED-routing as the conceptual center
- module-stack interpretation as final project identity
- purely temperature-based branch differentiation as enough
- adapter plasticity without explicit anchor-conditioned learning semantics

## 15. Clean architectural summary

If the theory is right, the next architecture must contain something functionally equivalent to:

1. Anchor detector
2. Anchor memory
3. Descendant linkage tracker
4. Contradiction monitor
5. Viability estimator
6. Revision mechanism
7. Anchor-aware branching/verifier layer
8. Anchor-conditioned self-learning layer

Not every component must be separate in code, but every function must exist somewhere in the system.

## 16. One-sentence design law

The next model should not merely attend over tokens.
It should detect, preserve, stress-test, revise, and learn around anchor spans according to the future structures they generate.
