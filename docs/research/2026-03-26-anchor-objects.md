# Anchor Objects — Elegant Working Definitions

Date: 2026-03-26
Status: active theory refinement memo
Related:
- `docs/research/2026-03-26-anchor-framework.md`
- `docs/research/2026-03-26-open-questions.md`
- `docs/research/2026-03-26-math-toy-cases.md`
- `docs/research/PROJECT_MAP.md`

## 1. Purpose

This note refines the most important unresolved objects in the anchor framework.
The goal is not to overformalize too early.
The goal is to arrive at definitions that are:
- conceptually clean
- mathematically usable
- empirically testable
- architecturally informative

The four objects treated here are:
- false anchor
- anchor viability
- contradiction pressure
- dead-end recognition point

## 2. Design principle

The framework should remain elegant.
That means each object should answer one distinct question:

- Anchor strength: how much future structure does this anchor organize?
- Anchor viability: can the current anchor still sustain a coherent future tree?
- Contradiction pressure: how much evidence is accumulating against that coherence?
- Dead-end recognition point: when should the model already know that the current anchor is no longer worth defending?

## 3. False anchor

### 3.1 Intuition
A false anchor is not merely a weak anchor.
It is an anchor that can be locally plausible and even initially productive, but whose induced future tree is nonviable relative to the task or future evidence.

### 3.2 Working definition
Let `a` be an anchor candidate with descendant structure `T(a)`.
We call `a` a false anchor if:

1. it receives enough local support to guide continuation,
2. it generates a nontrivial descendant tree,
3. but under future consistency pressure its tree becomes inferior or nonviable compared with the task's actual constraints or with a better competing anchor.

In compact form:

A false anchor is a locally supported but globally nonviable semantic root.

### 3.3 Important distinction
- Weak anchor: low influence, low tree mass, little downstream control.
- False anchor: real influence, real descendants, but wrong long-horizon commitments.

This distinction matters because false anchors are dangerous precisely because they are strong enough to reorganize continuation before failing.

## 4. Anchor viability

### 4.1 Intuition
Anchor viability answers:

> if the model keeps trusting this anchor, can the future still unfold coherently?

### 4.2 Working definition
For anchor `a` at time or depth `t`, let `V(a,t)` denote the degree to which the descendants currently induced by `a` remain jointly sustainable.

Interpretation:
- high `V(a,t)`: the anchor continues to support a coherent future tree
- medium `V(a,t)`: the anchor is under stress but still usable
- low `V(a,t)`: the anchor is close to collapse
- near-zero `V(a,t)`: the anchor is effectively dead-end

### 4.3 Minimal structural view
A clean first-order form is:

`V(a,t) = f(local support, descendant coherence, contradiction pressure)`

The exact function can be chosen later, but conceptually:
- local support pushes viability up
- coherent descendants push viability up
- contradiction pressure pushes viability down

### 4.4 Elegant interpretation
Anchor strength says how much future an anchor controls.
Anchor viability says whether that future is still livable.

## 5. Contradiction pressure

### 5.1 Intuition
Contradiction pressure is the accumulated evidence that the current anchor is forcing the future into tension, patching, or incompatibility.

It should not mean only formal logical contradiction.
It should also include structural strain.

### 5.2 Working definition
For anchor `a`, let `C(a,t)` denote the accumulated pressure against the continued validity of `a` up to time or depth `t`.

Sources of contradiction pressure may include:
- explicit logical inconsistency
- inability to satisfy an induced obligation
- increasing need for local repairs or exceptions
- branch divergence around anchor-sensitive descendants
- verifier confidence collapse
- retrospective superiority of an alternative anchor

### 5.3 Important idea
Contradiction pressure may begin before explicit contradiction appears.
That is what makes it useful.

A theory with only terminal contradiction is too late.
A good theory tracks pre-collapse strain.

### 5.4 Elegant interpretation
Contradiction pressure is the cost of continuing to believe the current anchor.

## 6. Dead-end recognition point

### 6.1 Intuition
The dead-end recognition point is not the final collapse.
It is the earliest point at which the model should already be able to infer that the current anchor is no longer worth preserving.

### 6.2 Working definition
For anchor `a`, define `t*` as a dead-end recognition point if:

1. the continuation has not yet fully collapsed,
2. but enough contradiction pressure has accumulated that a rational anchor-aware system should lower confidence, revise, or abandon `a`,
3. and delaying revision beyond `t*` predictably increases future repair cost or failure probability.

### 6.3 Distinction from collapse point
- Dead-end recognition point: earliest justified moment of revision
- Collapse point: later moment when failure becomes explicit or unavoidable

The gap between them is extremely important.
A strong system should widen this gap in the useful direction:
- early recognition
- late or avoided collapse

### 6.4 Elegant interpretation
A good model does not wait for the wall.
It notices that the road leads to the wall.

## 7. Relations among the four objects

These objects form a clean dependency chain:

1. An anchor gains local support and begins to generate descendants.
2. Descendants either remain coherent or begin to accumulate contradiction pressure.
3. Contradiction pressure reduces anchor viability.
4. Once viability falls far enough, a dead-end recognition point should appear.
5. If the model still refuses revision, the false anchor eventually reaches collapse.

In compact form:

`false anchor -> contradiction pressure -> viability drop -> dead-end recognition -> collapse (if uncorrected)`

## 8. First-order operational picture

A minimal future implementation could treat anchor state as:
- `A(a)` — current anchor score
- `Omega(a)` — semantic weight / strength
- `C(a,t)` — contradiction pressure
- `V(a,t)` — viability
- `state(a)` in {candidate, provisional, confirmed, decaying, dead-end}

Then a natural policy would be:
- high anchor score creates a candidate or provisional anchor
- sustained low contradiction pressure promotes it toward confirmed
- rising contradiction pressure lowers viability
- low viability plus sufficient pressure marks a dead-end recognition event
- revision or branch switch happens before full collapse

## 9. Clean research predictions

If these definitions are right, then we should observe:

1. Some anchors with high initial score later show sharply declining viability.
2. False anchors should often be distinguishable from weak anchors by having larger early tree mass but faster later viability collapse.
3. Contradiction pressure should become measurable before explicit failure.
4. Anchor-aware revision should outperform systems that react only to generic uncertainty.
5. Mathematics should exhibit especially clear dead-end recognition structure because descendant obligations are explicit.

## 10. What this clarifies for architecture

These definitions imply distinct architectural needs:

- anchor scorer: detect candidate anchors
- anchor memory: keep anchors alive across their semantic lifetime
- contradiction monitor: estimate pressure against current anchors
- viability tracker: decide whether an anchor still deserves trust
- revision mechanism: switch, branch, or downgrade false anchors before collapse

This is cleaner than treating all uncertainty as one scalar.

## 11. Minimal one-line definitions

For fast recall:

- False anchor: a locally supported but globally nonviable semantic root.
- Anchor viability: the degree to which an anchor can still sustain a coherent future tree.
- Contradiction pressure: accumulated evidence that the current anchor is forcing the future into tension or inconsistency.
- Dead-end recognition point: the earliest point where the model should already know the current anchor is no longer worth defending.

## 12. Next step

The next natural refinement is to connect these objects to the toy mathematical cases and annotate each case with:
- candidate false anchor
- observed contradiction pressure signals
- estimated viability drop
- earliest plausible dead-end recognition point
