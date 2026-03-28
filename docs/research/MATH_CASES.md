# Math Cases — ABPT

Date: 2026-03-27
Status: active canonical toy-case set
Source:
- `docs/research/2026-03-26-math-toy-cases.md`

## 🎯 Purpose

This file is the compact active set of math toy cases used to pressure-test the anchor framework.
It is not a benchmark suite yet.
It is a theory-validation substrate.

## 1. Universal vs existential root

### True anchor
- `for all n`

### Plausible false anchor
- `there exists n`

### Why it matters
A quantifier root changes the entire future tree.
If the root is wrong, later obligations become globally inconsistent.

### First contradiction point
The proof begins using witness-style reasoning where uniform reasoning is required.

### Dead-end recognition point
The continuation cannot maintain a single argument that covers every `n`.

## 2. Direct proof vs contradiction

### True anchor
- `proof by contradiction`

### Plausible false anchor
- direct constructive reading

### Why it matters
The proof regime itself becomes the anchor.
A wrong regime produces the wrong descendants.

### First contradiction point
The argument relies on assuming the negation, but the current tree does not support that move.

### Dead-end recognition point
The proof keeps patching local steps without producing a clean contradiction or direct implication.

## 3. Induction vs finite examples

### True anchor
- induction hypothesis

### Plausible false anchor
- repeated finite example checking

### Why it matters
A false finite-example root can look locally persuasive while failing to support the infinite case.

### First contradiction point
The argument never constructs the `k -> k+1` step.

### Dead-end recognition point
The proof has many local validations but no general mechanism.

## 4. Epsilon-delta vs intuitive closeness

### True anchor
- formal `epsilon-delta` regime

### Plausible false anchor
- informal closeness narrative

### Why it matters
The anchor determines whether the future should satisfy formal quantifier obligations or rhetorical approximation.

### First contradiction point
The proof needs quantified bounds but keeps producing descriptive language.

### Dead-end recognition point
The continuation cannot discharge the needed nested quantifier obligations.

## 5. Prime modulus vs generic modulus

### True anchor
- `mod p` where primality matters

### Plausible false anchor
- generic modulus treatment

### Why it matters
A wrong algebraic root may still look fluent for several steps and only later fail.

### First contradiction point
The continuation uses an invertibility step that is only valid under the prime-specific root.

### Dead-end recognition point
The argument cannot justify a required cancellation or field-like move.

## 6. O(1) vs O(n)

### True anchor
- constant-time access/update

### Plausible false anchor
- linear scan hidden under fluent prose

### Why it matters
This is the algorithmic version of a false anchor: a local story that creates the wrong global cost tree.

### First contradiction point
A later step implicitly iterates over state the method claimed to access directly.

### Dead-end recognition point
The claimed complexity can no longer be defended without reinterpreting the earlier algorithm.

## 7. Exact equality vs approximation

### True anchor
- exact identity/equality claim

### Plausible false anchor
- approximate equivalence

### Why it matters
The root determines whether downstream steps must preserve equality or can tolerate slack.

### First contradiction point
A later step requires equality-preserving substitution but earlier steps only supported approximation.

### Dead-end recognition point
The proof trajectory starts borrowing exact moves from an inexact root.

## 📌 What these cases are for

These cases are not for generic LM scoring.
They are for checking whether the future model can:
- detect the intended root anchor
- represent a plausible false anchor separately
- show growing contradiction pressure
- reach a dead-end recognition point before total failure

## One-sentence summary

The math cases are compact stress tests where one early anchor reading changes the entire future proof tree, making them ideal for testing false-anchor collapse and pre-collapse revision.
