# ABPT autoresearch Program

> Historical note: this program reflects the earlier module-stack research phase.
> The active theory pivot is now documented in `docs/research/2026-03-26-anchor-framework.md`.

## Goal
Test whether combining Attention Residuals, plastic layer, branch heads, and verifier
gives a small transformer (10-20M params) quality comparable to 2-5x larger baseline.

## Current research pivot (2026-03-26)

The current primary question is no longer only whether a module stack beats a baseline.
The deeper question is whether generation depends on anchor spans: context-emergent semantic units whose influence persists over long horizons and unfolds into dependency trees.

The current working framework has three linked mathematical layers:

1. Failure law
   - P_hall = 1 - (1 - r)^K
   - why tasks with many critical anchors become brittle

2. Detection law
   - A(a) = sigma(w1 * S_prior(a) + w2 * S_runtime(a))
   - how anchor spans may be detected at runtime

3. False-anchor collapse law
   - P_dead(a,d) ~= 1 - exp(-lambda * M(a,d))
   - how false anchors may generate dead-end trees that eventually collapse

This reframes the project:
- hallucination becomes anchor miss / misread / decay
- branches/verifier/plasticity must be reinterpreted through anchor theory
- future experiments should test true vs false anchors, anchor trees, and dead-end recognition points

## Architecture
Small transformer with:
- Attention Residuals (selective depth aggregation, replaces standard residuals)
- Plastic adapter layer (gradient-based, with decay and L2 regularization)
- 2 branch output heads (with diversity loss to prevent collapse)
- Verifier (entropy + agreement scoring to select best branch)

All modules togglable via config for clean ablation.

## Experiment Plan

### Phase 1: Baseline
1. Train baseline-0 (plain transformer) for 5 min, record val_bpb
2. Train baseline-1 (+ AttnRes) for 5 min, compare

### Phase 2: Individual Modules
3. Train baseline-2 (AttnRes + branches + verifier), compare
4. Train baseline-3 (AttnRes + plastic), compare

### Phase 3: Full Model
5. Train full model (all modules), compare with all baselines
6. Record: val_bpb, branch_diversity, adaptation_gain, compute_overhead

### Phase 4: Synthetic Tasks
7. Test on ambiguous-choice tasks (verifier should help)
8. Test on temporary-rule tasks (plastic layer should help)
9. Test on style-adaptation tasks (plastic + verifier should help)

### Phase 5: Anchor-theory experiments (new active direction)
10. Build toy mathematical cases with explicit true vs false anchors
11. Trace anchor descendants and locate the first contradiction point
12. Search for dead-end recognition points before final collapse
13. Test whether anchor-aware correction beats generic uncertainty-based correction

## Metrics
- val_bpb (primary, old module-stack phase)
- task accuracy (per synthetic task)
- branch diversity (cosine distance between branch outputs)
- adaptation gain (plastic vs no-plastic on temporary-rule)
- compute overhead (FLOPs ratio vs baseline-0)
- anchor detection precision/recall (future)
- dead-end recognition lead time (future)
- anchor-tree contradiction pressure (future)

## Constraints
- 5 min per experiment on T4
- Keep changes minimal between experiments
- Log all results to results/ directory
- If val_bpb improves by < 0.5% over baseline, module is not worth the overhead
- Do not lock major code architecture until the anchor framework stabilizes
