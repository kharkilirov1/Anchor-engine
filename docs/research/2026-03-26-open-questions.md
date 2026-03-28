# Anchor Framework — Open Questions and Research Gaps

Date: 2026-03-26
Status: active unresolved questions memo
Related: `docs/research/2026-03-26-anchor-framework.md`

## 1. Purpose

This note captures what is still unresolved in the anchor framework so the theory can continue to evolve without losing precision.

The goal is not to freeze the architecture too early.
The goal is to preserve:
- what already looks stable
- what is still only a hypothesis
- what must be tested before major redesign

## 2. What currently looks stable

These points now look like the strongest working assumptions:

1. Not all context is equal in long-horizon influence.
2. The correct unit is likely an anchor span, not an isolated token.
3. Hallucination is often better explained as anchor miss / misread / decay than as generic uncertainty alone.
4. Mathematics is likely especially sensitive because anchor density and dependency depth are high.
5. A false anchor may generate a dead-end tree rather than fail immediately.
6. Inference-time learning may need to focus on anchors rather than diffuse token-level updates.

These are still empirical claims, but they are the most stable parts of the current theory.

## 3. Primary unresolved questions

### 3.1 What exactly closes a span?
The theory says anchors are spans, but the operational boundary is still unclear.

Open possibilities:
- purely contiguous n-gram completion
- latent semantic closure event in hidden state space
- syntax-aware closure
- task-specific closure

Open question:
What is the best runtime criterion for saying "this anchor span has formed"?

### 3.2 Is the true structure a tree or a DAG?
The current formulation uses anchor trees because they are intuitive and analytically useful.
But future dependency structure may be richer:
- one descendant may depend on multiple anchors
- several anchors may merge into one later semantic obligation
- some nodes may have multiple parents

Working question:
Should the mature theory use anchor trees, anchor forests, or anchor dependency DAGs?

### 3.3 What is the right notion of semantic lifetime?
The framework says an anchor should live as long as it matters.
But that still needs operational definition.

Possible interpretations:
- fixed horizon length
- learned decay schedule
- survival until contradiction pressure crosses a threshold
- survival until the tree mass it controls has mostly unfolded

Open question:
How should semantic lifetime be measured and updated online?

### 3.4 What distinguishes a strong anchor from a super-anchor?
Current theory suggests:
- strength = immediate causal effect
- tree mass = total downstream influence
- depth/breadth = structural influence

But the boundary between anchor and super-anchor is not formalized.

Open question:
Is a super-anchor simply a high-weight anchor, or an anchor that changes the whole mode of reasoning?

### 3.5 What is a false anchor mathematically?
Current intuition:
- false anchors can be locally plausible
- they may generate descendants before failure
- contradiction accumulates later

Still unresolved:
- must a false anchor be globally wrong?
- can it be partially correct but overcommitted?
- is falsehood defined by retrospective inferiority to another anchor?

Open question:
What exact formal criterion separates a false anchor from a weak-but-valid anchor?

## 4. Dead-end recognition: the key unresolved mechanism

This is currently one of the most important open problems.

The theory now suggests there may exist a point before full collapse where the model could already infer that the current anchor is leading to a dead end.

This point is not yet formalized.

Candidate signals:
- growing contradiction pressure in descendants
- branch divergence spike
- verifier confidence collapse
- rising repair cost / anchor debt
- retrospective superiority of an alternative anchor

Open question:
What is the earliest reliable signal that a false anchor has become nonviable?

Related question:
Can the model revise an anchor before the full tree collapses?

## 5. Anchor-based self-learning: strongest promise, highest risk

The framework now points toward anchors as units of inference-time credit assignment.
That is a major theoretical opportunity.
It is also dangerous.

### 5.1 Why it is promising
Anchors may tell the model:
- what to learn from
- when to adapt
- where to localize updates
- which errors matter most

### 5.2 Why it is dangerous
A false anchor may trigger the wrong self-update and reinforce the wrong semantic frame.
This can create self-amplifying errors.

Open questions:
- should learning happen only for confirmed anchors?
- should provisional anchors decay unless validated?
- should false-anchor collapse itself become a supervised learning event?
- what kind of updates should be reversible?

## 6. Relationship to current ABPT modules

The current codebase still reflects the earlier module-stack phase.
The mapping into the new theory is only partially resolved.

### 6.1 AttnRes
Looks useful as anti-dilution machinery.
Unresolved question:
Is depth anti-dilution sufficient, or does anchor preservation require explicit anchor memory beyond layer mixing?

### 6.2 Branches
Current likely reinterpretation:
competing anchor readings.
Unresolved question:
Should branching happen only around ambiguous anchors instead of general uncertainty?

### 6.3 Verifier
Current likely reinterpretation:
anchor-reading selector and contradiction monitor.
Unresolved question:
Can verifier become the main detector of dead-end recognition points?

### 6.4 Plasticity
Current likely reinterpretation:
anchor-conditioned self-learning.
Unresolved question:
What should actually be updated: anchor memory, local adapters, span representations, or routing policy?

### 6.5 Stage B routing
Current code uses equilibrium deviation and route buckets.
Unresolved question:
Is generic ED-based routing only a temporary proxy, with anchor-event routing as the real target?

## 7. What must be tested before large code redesign

### 7.1 True vs false anchor toy cases
Build minimal cases, especially in mathematics, where:
- one anchor is valid
- one anchor is plausible but false
- descendant structure can be traced explicitly

### 7.2 Dead-end recognition point
For each false-anchor case, identify:
- root anchor
- first descendants
- first contradiction
- earliest point where the dead end should have been detectable
- final collapse point

### 7.3 Anchor density hypothesis
Test whether anchor-heavy tasks fail faster than anchor-light tasks under controlled perturbation.

### 7.4 Anchor lifetime hypothesis
Test whether performance improves when important spans are explicitly preserved for longer horizons.

### 7.5 False-anchor correction hypothesis
Test whether a system that can revise anchors outperforms one that only measures generic uncertainty.

## 8. Suggested notation cleanup

The framework should avoid symbol overload before it grows further.

Suggested symbols:
- Omega(a): semantic weight / anchor strength
- A(a): anchor detection score
- M(a,d): anchor-tree mass
- V(a,d): anchor viability
- C(a,d): contradiction pressure
- P_hall: hallucination probability
- P_dead(a,d): dead-end probability

## 9. Immediate next theory tasks

1. Formalize anchor viability and contradiction pressure.
2. Define false anchor more rigorously.
3. Define dead-end recognition point operationally.
4. Decide whether the mature object is tree, forest, or DAG.
5. Build the first mathematical toy cases by hand before coding.
6. Only then decide which parts of ABPT should survive into the next architecture.

## 10. Honest status

Current status should be stated precisely:
- the framework is conceptually strong
- the formulas are first-order and internally coherent
- the theory is not yet empirically validated
- several central objects are still under definition
- major implementation choices should remain provisional until the open questions are narrowed
