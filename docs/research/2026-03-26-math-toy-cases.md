# Mathematical Toy Cases for Anchor Theory

Date: 2026-03-26
Status: active research test set draft
Related:
- `docs/research/2026-03-26-anchor-framework.md`
- `docs/research/2026-03-26-open-questions.md`
- `docs/research/PROJECT_MAP.md`

## 1. Purpose

This document defines small, hand-analyzable mathematical cases for testing the anchor framework.
The goal is not benchmark performance yet.
The goal is to test whether the theory gives clear predictions about:
- true anchors
- plausible false anchors
- descendant structure
- first contradiction points
- dead-end recognition points

These toy cases are meant to be analyzed conceptually first, and only later translated into model-facing evaluation tasks.

## 2. How to read each case

Each case is structured as:
- Task
- True anchor
- Plausible false anchor
- Descendant tree sketch
- First contradiction point
- Dead-end recognition point
- What should happen if the framework is right

The key distinction is:
- a true anchor generates a viable future tree
- a false anchor can still generate early descendants, but later accumulates contradiction pressure

## 3. Case A — Universal quantifier mistaken for existential quantifier

### Task
Reason about a statement of the form:
- `For all n in N, P(n) holds`

### True anchor
- `for all n`
- semantic role: universal commitment

### Plausible false anchor
- `there exists n`
- or an example-based reading: one successful instance is enough

### Descendant tree sketch
True-anchor descendants:
1. obligation to reason for arbitrary `n`
2. need for generic argument structure
3. no dependence on a special chosen instance
4. conclusion must hold uniformly

False-anchor descendants:
1. choose a convenient witness or example
2. verify `P(n0)` for one or a few values
3. overgeneralize from witness to universal claim

### First contradiction point
The proof needs a generic arbitrary-variable step, but the false tree only supports witness-based reasoning.

### Dead-end recognition point
As soon as the argument's next step requires preserving arbitrariness and the current branch already committed to a specific example.

### Framework prediction
A false existential reading may look locally plausible for several steps, but eventually its tree cannot satisfy the universal obligation.

## 4. Case B — Direct proof vs proof by contradiction

### Task
The theorem requires a contradiction-based proof regime.

### True anchor
- `proof by contradiction`
- semantic role: assume negation, derive contradiction

### Plausible false anchor
- `direct proof`
- or a generic local manipulation without switching proof regime

### Descendant tree sketch
True-anchor descendants:
1. negate target claim
2. derive consequences under negation
3. connect consequences to impossibility
4. discharge negation by contradiction

False-anchor descendants:
1. keep target claim positive
2. try to manipulate expressions directly
3. accumulate steps that never connect to contradiction structure

### First contradiction point
The argument introduces or requires use of the negated assumption, but the false branch never created that object.

### Dead-end recognition point
When the reasoning starts referring to a contradiction target that does not exist in the current branch state.

### Framework prediction
A wrong proof-mode anchor should generate an initially coherent but structurally incompatible tree.

## 5. Case C — Induction anchor vs finite-example anchor

### Task
Prove a proposition over natural numbers by induction.

### True anchor
- `induction on n`
- or `assume statement holds for k`
- semantic role: induction schema

### Plausible false anchor
- `check first few cases`
- semantic role: empirical pattern confirmation mistaken for proof skeleton

### Descendant tree sketch
True-anchor descendants:
1. base case
2. induction hypothesis
3. induction step `k -> k+1`
4. universal conclusion over all `n`

False-anchor descendants:
1. verify `n = 1, 2, 3`
2. detect pattern
3. infer general truth from pattern continuation

### First contradiction point
The proof requires an induction step using the hypothesis, but the false tree has only examples and no transferable bridge.

### Dead-end recognition point
When the next justified move requires expressing dependence on an arbitrary `k` rather than a finite checked set.

### Framework prediction
Example-based reasoning should often act as a strong but false anchor in induction tasks.

## 6. Case D — Epsilon-delta anchor vs intuitive closeness anchor

### Task
Show a limit statement using formal epsilon-delta reasoning.

### True anchor
- `for every epsilon > 0 there exists delta > 0`
- semantic role: quantified control structure for limits

### Plausible false anchor
- `the graph looks close`
- semantic role: informal geometric intuition mistaken for formal proof anchor

### Descendant tree sketch
True-anchor descendants:
1. choose arbitrary `epsilon`
2. solve for `delta` in terms of `epsilon`
3. propagate inequality constraints
4. conclude formal limit statement

False-anchor descendants:
1. talk about values becoming close
2. use approximate intuition
3. skip the quantifier dependencies between `epsilon` and `delta`

### First contradiction point
The argument must produce `delta` as a function of `epsilon`, but the false tree only contains informal closeness language.

### Dead-end recognition point
When the proof requires explicit quantified dependency and no symbolic mechanism exists in the branch.

### Framework prediction
In formal analysis tasks, intuitive proximity anchors may be locally smooth but structurally nonviable.

## 7. Case E — Prime modulus vs generic integer modulus

### Task
Reason in modular arithmetic where primality matters.

### True anchor
- `mod p, where p is prime`
- semantic role: field-like algebraic permissions under prime modulus

### Plausible false anchor
- `mod n` for generic `n`
- semantic role: treating composite modulus as if the same inverses and cancellations always work

### Descendant tree sketch
True-anchor descendants:
1. use invertibility of nonzero classes
2. cancel safely under prime assumptions
3. derive field-structured consequences

False-anchor descendants:
1. perform cancellations too freely
2. assume inverses where they may not exist
3. derive algebraic consequences that depend on primality but were never justified

### First contradiction point
A cancellation or inverse step is required that is only valid in the prime-modulus regime.

### Dead-end recognition point
At the first operation whose legality depends on field structure rather than generic ring structure.

### Framework prediction
This is a clean case where a false anchor can remain hidden until one specific algebraic move exposes it.

## 8. Case F — O(1) anchor vs O(n) anchor in algorithm analysis

### Task
Analyze running time or feasibility of a method.

### True anchor
- `requires scan over n elements`
- semantic role: linear-cost structural commitment

### Plausible false anchor
- `constant-time lookup`
- semantic role: overoptimistic local operation mistaken for whole-algorithm cost anchor

### Descendant tree sketch
True-anchor descendants:
1. identify full traversal requirement
2. count operations per element
3. obtain linear complexity

False-anchor descendants:
1. focus on one cheap operation
2. ignore hidden traversal or preprocessing
3. claim O(1)-style complexity too early

### First contradiction point
A later step reveals unavoidable dependence on all elements or on input size.

### Dead-end recognition point
When maintaining the O(1) story requires ad hoc exceptions or hidden assumptions not present at the root.

### Framework prediction
This case is useful because the false anchor often creates growing anchor debt rather than immediate symbolic contradiction.

## 9. Case G — Equality anchor vs approximation anchor

### Task
Prove an exact algebraic identity.

### True anchor
- `exact equality`
- semantic role: every transformation must preserve identity exactly

### Plausible false anchor
- `close enough approximation`
- semantic role: replacing exact reasoning with asymptotic or numerical intuition

### Descendant tree sketch
True-anchor descendants:
1. exact transformations
2. exact cancellation/factorization
3. exact target equality reached

False-anchor descendants:
1. use approximate simplification
2. replace equality with near-equality intuition
3. continue as though exact proof obligations were preserved

### First contradiction point
The argument requires exact equality preservation, but the false tree has already introduced approximation error.

### Dead-end recognition point
When the branch must conclude a symbolic identity from steps that only justify closeness.

### Framework prediction
This distinguishes formal proof anchors from heuristic numerical anchors.

## 10. Shared patterns across the toy set

These cases suggest several recurring research motifs:

1. False anchors are often not weak; they are locally compelling.
2. A false anchor may generate several valid-looking descendants before failure.
3. Failure is often delayed until a structurally indispensable obligation appears.
4. Dead-end recognition may occur before full collapse if contradiction pressure is tracked.
5. Mathematics is a strong testbed because descendant obligations are explicit and hard to fake indefinitely.

## 11. Candidate signals to log in future experiments

For each toy case, future model analysis should try to log:
- anchor score over time
- branch divergence around anchor-sensitive steps
- verifier confidence over competing anchor readings
- contradiction pressure proxy
- repair cost / anchor debt
- retrospective superiority of alternative anchor
- distance between dead-end recognition point and final collapse point

## 12. Immediate follow-up

These toy cases should next be used to define:
1. a formal false-anchor criterion
2. an operational dead-end recognition point
3. the first anchor-specific evaluation tasks
4. architecture requirements for anchor memory and anchor revision
