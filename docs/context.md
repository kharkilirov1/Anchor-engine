# ABPT v2 — Architecture Context

> Historical note: this file captures the original ABPT v2 / Stage A-Stage B framing.
> The current active theoretical center has shifted toward an anchor-centric framework.
> Read `docs/research/2026-03-26-anchor-framework.md` first for the latest research direction.

Source: "ABPT v2 complete.docx" — concept paper / research proposal / technical design spec.

## Current theoretical pivot (2026-03-26)

The project is no longer best understood as "AttnRes + branches + verifier + plasticity" as an end in itself.
The current core hypothesis is that generation quality depends on detecting, preserving, revising, and learning around anchor spans:
- context-emergent semantic units with long-horizon causal influence
- roots of future dependency trees
- primary units behind hallucination, correction, and potentially inference-time self-learning

In this framing:
- AttnRes is a useful borrowed anti-dilution mechanism, not the conceptual core
- branches may become competing anchor interpretations
- verifier may become an anchor-reading selector / contradiction monitor
- plasticity may become anchor-conditioned self-learning
- Stage B routing likely needs to be re-grounded around anchor events rather than generic ED alone

## Central Hypothesis

Combining six mechanisms in one architecture — selective depth aggregation (Attention Residuals),
stable long-term core, plastic short-term layer, multiple output branches, verification module,
and limited inference-time adaptation — a small model can match quality of 2-5x larger models
on tasks where doubt, local adaptation, and choosing between close patterns matter.

## MVP Stage A — Separate Modules

| Module | Description |
|--------|-------------|
| Stable Core + AttnRes | Small transformer backbone, AttnRes replaces standard residual connections |
| 2 Branch Heads + Diversity Loss | Horizontal branching, diversity loss prevents collapse |
| Verifier | Entropy + inter-branch agreement + confidence score |
| Plastic Layer | Gradient-based adapter, small LR, L2 to init, decay |

## MVP Stage B — Unified (future)

| Module | Description |
|--------|-------------|
| Equilibrium Signal | Based on LayerNorm running stats, near-zero overhead |
| Adaptive Routing | Forward/backward/branch routing based on ED |
| SoA scatter/gather | Token grouping by route, dense batches |
| Day/Night Cycle | GPU computes Day Phase, CPU restructures Night Phase |
| Token Energy Budget | Per-token compute limit, 3-4 route buckets |

## Key Design Decisions

1. AttnRes: each layer attends to all previous layer outputs (softmax weights) instead of simple residual addition
2. Plastic layer: only adapter weights update at inference, backbone frozen
3. Branch diversity: different temperature/attention masks per branch head
4. Verifier scoring: weighted combination of entropy, agreement, consistency
5. All modules togglable via config flags for clean ablation

## Four Selectivity Dimensions

| Dimension | Mechanism | Problem Solved |
|-----------|-----------|---------------|
| Depth (vertical) | Attention Residuals | PreNorm dilution |
| Width (horizontal) | Branch Router + Verifier | Premature collapse to first pattern |
| Time (adaptive) | Plastic Layer | Static inference |
| Direction (routing) | Adaptive Routing + Equilibrium Signal | Fixed compute path |

## Model Sizes

| Profile | Params | Purpose |
|---------|--------|---------|
| Toy | 3-10M | Logic verification, CPU/Kaggle T4 |
| Research Small | 10-20M | Main experiments, Kaggle/Colab T4 |
| Research Mid | 20-40M | Effect confirmation (only after positive signals) |

## Baselines for Ablation

| Model | Contains |
|-------|----------|
| Baseline-0 | Plain small transformer |
| Baseline-1 | Transformer + AttnRes |
| Baseline-2 | AttnRes + 2 branches + verifier |
| Baseline-3 | AttnRes + plastic layer |
| Target | AttnRes + branches + verifier + plastic |

## Success Metrics

- val_bpb (validation bits per byte)
- accuracy / task success
- consistency (inter-branch agreement)
- adaptation gain (plastic layer benefit)
- branch diversity (anti-collapse metric)
- parameter efficiency ratio vs baseline
- compute overhead vs baseline

## Key Risks

| Risk | Probability | Countermeasure |
|------|------------|----------------|
| Branch collapse | 80% | Diversity loss, different temps/masks |
| Plastic layer noise | — | Decay, thresholds, L2 regularization |
| Verifier overhead not justified | 60% | Toggleable, lightweight architecture |
| AttnRes + modules conflict | — | Sequential addition and testing |
| Combination lacks synergy | 70% | Ablation reveals which module helps |

## References

- Attention Residuals: Kimi Team, 2026 (arXiv:2603.15031)
- Test-Time Training: Sun et al., 2024
- Complementary Learning Systems: McClelland et al., 1995
- Axicor (bio-plausible spiking nets): H4V1K-dev, 2026
- autoresearch: Karpathy, 2026
