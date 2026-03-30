# Project Map — ABPT

Date: 2026-03-30
Status: active navigation map

## 1. What this project is right now

ABPT is currently in a theory-first research phase.
The active conceptual center is the anchor-centric framework, not the historical AttnRes + branches + verifier + plasticity stack by itself.

In current terms:
- anchor spans are context-emergent semantic control units
- hallucination is treated as anchor miss / misread / decay
- false anchors may generate dead-end trees
- revision should happen before collapse, not after it

## 2. Active read order

Read in this order:

1. `docs/research/ANCHOR_TREE_IMPLEMENTATION_PLAN.md`
   - active implementation plan for tree-level proposal/revision control
2. `docs/research/ANCHOR_THEORY.md`
   - compact canonical theory memo
3. `docs/research/MATH_CASES.md`
   - compact toy-case set for false-anchor reasoning
4. `docs/research/ARCHITECTURE_V1.md`
   - compact canonical architecture memo
5. `docs/research/SUBSYSTEM_INTERFACES.md`
   - last pre-code interface layer
6. `docs/research/QWEN_INTEGRATION_PLAN.md`
   - approved integration plan for testing the anchor engine on top of a stronger pretrained model
7. `docs/context.md`
   - bridge from historical ABPT framing to the anchor-centric pivot
8. `program.md`
   - active research program

## 3. Active documents

### Active core
- `docs/research/ANCHOR_TREE_IMPLEMENTATION_PLAN.md`
- `docs/research/ANCHOR_THEORY.md`
- `docs/research/MATH_CASES.md`
- `docs/research/ARCHITECTURE_V1.md`
- `docs/research/SUBSYSTEM_INTERFACES.md`
- `docs/research/QWEN_INTEGRATION_PLAN.md`
- `docs/context.md`
- `program.md`

### Supporting dated notes
- `docs/research/2026-03-26-anchor-framework.md`
- `docs/research/2026-03-26-open-questions.md`
- `docs/research/2026-03-26-anchor-objects.md`
- `docs/research/2026-03-26-math-toy-cases.md`
- `docs/research/2026-03-26-architecture-requirements.md`
- `docs/research/2026-03-26-keep-rewrite-retire.md`
- `docs/research/2026-03-27-next-architecture-sketch.md`
- `docs/research/2026-03-27-v1-implementation-spec.md`

### Historical note
- some older design assumptions still survive in scattered files
- the public repo snapshot intentionally excludes most generated artifacts and legacy benchmark archives

## 4. Current code meaning

The current codebase still mostly reflects the earlier module-stack phase.
It should not yet be mistaken for a full implementation of the anchor-centric theory.

### Current interpretation
- backbone remains useful
- AttnRes remains useful but borrowed
- branches/verifier/plasticity are candidate mechanisms, not the center
- Stage B routing prototype is not the new architecture base

## 5. Immediate next step

The immediate technical direction is now narrower:
1. implement the anchor-tree / tree-matching Sprint 1 core
2. use tree coverage / drift before adding more generation interventions
3. only then reconnect proposal/revision to inference on top of the stronger pretrained overlay

## 6. One-sentence summary

ABPT now has a compact active theory-and-design surface and is ready to move from documentation into a first anchor-centric implementation.
