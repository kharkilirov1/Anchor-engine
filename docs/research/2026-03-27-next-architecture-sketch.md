# Next Architecture Sketch — Anchor-Centric ABPT

Date: 2026-03-27
Status: active conceptual architecture sketch
Related:
- `docs/research/2026-03-26-anchor-framework.md`
- `docs/research/2026-03-26-anchor-objects.md`
- `docs/research/2026-03-26-architecture-requirements.md`
- `docs/research/2026-03-26-keep-rewrite-retire.md`
- `docs/research/PROJECT_MAP.md`

## 1. Purpose

This note sketches the next architecture at the subsystem level.
It is intentionally more concrete than the requirements memo, but still not yet implementation code.

The question answered here is:

> If the anchor framework is correct, what should the next model look like in broad structural terms?

## 2. Design law

The next model should not merely pass tokens through attention.
It should:
- detect anchors
- keep active anchors alive
- track which future structure depends on them
- monitor contradiction pressure
- revise false anchors before collapse
- learn selectively from anchor events

## 3. High-level picture

A useful first sketch is a two-layer system:

1. **Flow layer**
   - standard contextual processing over tokens
   - backbone transformer + optional AttnRes

2. **Anchor layer**
   - explicit or semi-explicit management of anchor objects
   - anchor detection, memory, viability, revision, and learning

One short slogan:

- attention carries the stream
- anchor layer governs the long-horizon semantic regime

## 4. Proposed subsystem stack

### 4.1 Backbone field
Base sequence processor.

Likely input:
- token embeddings
- positional information
- optional active-anchor conditioning

Likely candidate basis:
- current transformer backbone
- optional AttnRes for anti-dilution

Role:
- produce contextual states
- provide the substrate from which anchors emerge
- remain general-purpose rather than anchor-specialized by itself

### 4.2 Anchor detector
Consumes recent hidden-state evolution and proposes anchor candidates.

Inputs:
- current hidden states
- recent window of hidden states
- optional prior span map

Outputs:
- anchor candidates
- anchor score `A(a)`
- candidate span boundaries or span hypotheses
- candidate semantic weight estimate

Role:
- detect when a context-emergent unit closes or sharply changes semantic mode
- turn diffuse token flow into candidate semantic objects

### 4.3 Anchor memory
Stores currently active anchors.

Per-anchor fields should minimally include:
- anchor representation
- anchor score
- semantic weight
- state: candidate / provisional / confirmed / decaying / dead-end
- lifetime estimate
- local support summary
- contradiction pressure estimate
- viability estimate

Role:
- preserve anchors according to semantic lifetime rather than pure recency
- allow multiple anchors to coexist
- provide a stable layer above raw token flow

### 4.4 Descendant tracker
Tracks which future states or obligations appear to depend on which anchors.

Inputs:
- active anchors
- new hidden states
- new anchor-sensitive events

Outputs:
- root-descendant links
- local descendant graph/tree structure
- per-anchor descendant mass updates

Role:
- support the notion of anchor tree mass
- expose which futures are being governed by which anchors
- make false-anchor collapse observable rather than mysterious

### 4.5 Contradiction monitor
Tracks pressure against active anchors before final failure.

Inputs may include:
- verifier disagreement
- branch divergence
- failed descendant obligations
- local repair cost / anchor debt
- explicit logical contradiction proxies
- retrospective superiority of alternative anchor branches

Outputs:
- contradiction pressure `C(a,t)` for each active anchor

Role:
- detect pre-collapse strain
- provide the stress signal that viability must respond to

### 4.6 Viability tracker
Computes whether an anchor is still worth defending.

Inputs:
- anchor score
- descendant coherence
- contradiction pressure
- recent support trajectory

Outputs:
- viability `V(a,t)`
- anchor state transitions

Role:
- separate strong anchors from surviving anchors
- decide whether to keep, downgrade, branch, or revise

### 4.7 Revision controller
Acts when a dead-end recognition point is reached.

Possible actions:
- downgrade anchor state
- branch alternative anchor readings
- switch to an alternative anchor root
- lower anchor control weight
- block consolidation of a dead-end anchor

Role:
- turn dead-end recognition into action
- prevent false anchors from continuing unchallenged

### 4.8 Anchor-aware branching
Generates competing anchor interpretations only when justified.

Difference from old branches:
- not generic multi-head guessing
- specifically used for ambiguous anchors or revision events

Role:
- allow provisional competition between anchor readings
- avoid full-graph branching everywhere

### 4.9 Anchor verifier / arbiter
Scores competing anchor readings and their descendant futures.

Difference from old verifier:
- not just entropy/agreement on logits
- should compare contradiction resilience and descendant coherence

Role:
- choose which anchor reading deserves trust
- support consolidation or revision

### 4.10 Anchor-conditioned plasticity
Local online learning around anchor events.

Trigger candidates:
- anchor novelty
- anchor confirmation
- anchor misread evidence
- anchor decay evidence
- false-anchor collapse

Role:
- adjust the model selectively around high-value semantic objects
- make self-learning structured rather than diffuse

## 5. Minimal end-to-end loop

A first clean inference loop could be:

1. Backbone processes next token/context.
2. Anchor detector proposes or updates anchor candidates.
3. Anchor memory updates states and lifetimes.
4. Descendant tracker attaches new obligations/futures to active anchors.
5. Contradiction monitor updates pressure.
6. Viability tracker updates anchor health.
7. If needed, revision controller triggers branch/revise/downgrade.
8. Verifier chooses among competing anchor readings.
9. Plasticity updates only if an anchor event justifies it.
10. Generation continues under the revised active-anchor set.

## 6. State machine sketch

A minimal anchor state machine:

- candidate
  - newly detected, low commitment
- provisional
  - influential, but still under trial
- confirmed
  - survived enough future pressure
- decaying
  - influence remains but is fading
- dead-end
  - no longer worth defending; must be revised or retired

This state machine is likely more important than many low-level implementation choices.

## 7. Where old ABPT pieces fit

### Keep with little change
- transformer backbone
- optional AttnRes

### Keep concept, change role
- branches -> anchor interpretation branches
- verifier -> anchor arbiter
- plasticity -> anchor-conditioned self-learning

### Rebuild around new objects
- Stage B routing
- equilibrium-based control logic
- any selective-compute policy that does not explicitly reference anchors

## 8. Preferred implementation philosophy

The first serious implementation of this sketch should prefer:
- explicit, inspectable anchor objects over purely implicit hope
- conservative revision logic over aggressive self-editing
- provisional memory over immediate consolidation
- narrow, local updates over global test-time drift

The first version does not need to be elegant in code, but the conceptual structure should remain elegant.

## 9. Strong simplification for v1

A practical v1 should not try to solve everything at once.
It can intentionally simplify to:

1. anchor detector
2. anchor memory with candidate/provisional/confirmed states
3. contradiction monitor
4. revision controller
5. optional simple branch verifier

That is already enough to test the core thesis.

Descendant tracking can begin as a coarse approximation rather than a perfect semantic graph.
Plasticity can initially be shallow and local.

## 10. Clean boundary between theory and code

At this stage, code should still be downstream of theory.
So this document should be treated as:
- more concrete than abstract theory
- less concrete than an implementation spec

It is a bridge document.

## 11. One-sentence architecture sketch

The next ABPT should be a transformer-based flow model augmented with an anchor layer that detects semantic roots, stores them by lifetime, tracks their descendants, measures contradiction pressure, revises false anchors before collapse, and learns selectively from anchor events.
