# ABPT Project Thesis and Kill Criteria

Date: 2026-04-11  
Status: active evaluation memo

## Main thesis

The project is **not** primarily about making optimization easier.
The main thesis is:

> **A non-uniform architecture with explicit anchor-state control and/or heterogeneous motif geometry can organize computation better than a plain small transformer under comparable budget.**

“Better” here means:

- more faithful handling of stable semantic structure,
- less destructive interference between functions,
- better capability-per-parameter on the domains that actually require those structures.

---

## Secondary theses

### Thesis B — trainability is a consequence, not the center

If the architecture is genuinely better organized, then we may also observe:

- faster convergence,
- lower sensitivity to initialization,
- smoother optimization,
- lower need for fragile tricks.

But these are **secondary signals**, not the core claim.

### Thesis C — domain match matters

Different domains reward different inductive biases.

- symbolic / regime-structured domains should favor explicit anchor logic,
- broad natural text should favor strong generic flow modeling,
- heterogeneous geometry may help both, but not equally.

The goal is therefore not “win everywhere immediately,” but to understand:

> **which architectural bias helps on which domain, and why.**

### Thesis D — FOG and Anchor play different roles

- **FOG** = geometry of computation,
- **Anchor** = object layer of reasoning and revision.

They should not be judged as mutually exclusive replacements by default.

---

## What counts as success

### Success tier 1 — structural success

The architecture exposes signals that a plain transformer does not explicitly expose:

- anchor creation,
- contradiction pressure,
- viability trajectories,
- revision events,
- function separation / motif specialization.

### Success tier 2 — empirical success on matched domains

The architecture beats or strongly matches baseline on domains where its inductive bias is appropriate:

- anchor-synthetic,
- symbolic consistency tasks,
- controlled revision / contradiction tasks.

### Success tier 3 — competitive spillover to natural text

The architecture remains competitive on natural text:

- not necessarily best immediately,
- but close enough that the extra structure is not pure overhead.

---

## What does not count as success

- easier optimization alone,
- prettier curves without better capability,
- synthetic-only wins with severe natural-text collapse,
- opaque complexity without explicit runtime signals,
- architecture that only works when routing/controller logic is effectively bypassed.

---

## Kill criteria

These are the conditions under which a direction should be downgraded or retired.

### Kill criterion 1 — no structural signal

If a proposed module does not produce distinct, inspectable runtime behavior,
it is not testing the thesis and should be removed.

### Kill criterion 2 — no matched-domain gain

If a module does not outperform or clearly improve matched synthetic domains after fair-budget runs,
it is probably not expressing the intended bias.

### Kill criterion 3 — natural-text tax too large

If the module remains materially worse on TinyStories-like data after longer runs
without giving a strong structural advantage elsewhere, it is not worth keeping.

### Kill criterion 4 — controller collapse

If the controller/routing mechanism only looks good while effectively inactive,
the result is invalid.

### Kill criterion 5 — complexity without leverage

If a module increases parameters / compute / code complexity but does not improve:

- matched-domain performance,
- structural interpretability,
- or cross-domain competitiveness,

it should be demoted.

---

## Current project stance

Given the current evidence:

- **Anchor** remains the main research direction.
- **FOG** is the preferred candidate for the flow substrate.
- **Old Stage B ED-routing** is no longer treated as the architectural center.

So the next architecture should be evaluated with this rule:

> **Anchor is the object-level hypothesis.  
> FOG is the flow-level hypothesis.  
> Optimization ease is evidence only if it accompanies better computation.**

---

## Immediate evaluation protocol

For each major iteration:

1. run equal-budget comparison,
2. inspect matched-domain performance,
3. inspect natural-text competitiveness,
4. inspect structural diagnostics,
5. decide: keep, reinterpret, or retire.

That keeps the project anchored to the scientific question instead of drifting into pure benchmark chasing or pure optimization chasing.
