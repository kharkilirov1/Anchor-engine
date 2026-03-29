# Qwen Integration Plan

Date: 2026-03-28  
Status: approved next-step plan

## Goal

Test the anchor engine on top of a stronger pretrained model instead of continuing to judge it only through small custom backbones.

The immediate target is:

- `Qwen2.5-1.5B`

The purpose is not to rewrite Qwen or claim a new foundation model. The purpose is to check whether anchor diagnostics and control become more meaningful when the underlying model already has a substantially richer world model.

## Why this direction

Current evidence suggests:

- the small anchor prototype is trainable;
- anchor health metrics can improve;
- proposal/revision remains weak during training;
- one plausible reason is that the base model is too weak or too undertrained for contradiction-sensitive anchor logic to fully matter.

This motivates a different test:

> keep a strong pretrained LM fixed, and attach the anchor engine as an overlay.

## Design principle

Do **not** start with full finetuning or architectural surgery inside Qwen.

Start with:

- inference-time overlay
- frozen pretrained backbone
- hidden-state extraction
- anchor diagnostics on top

This is the cheapest and cleanest falsification path.

## Phase 1 — Qwen anchor overlay MVP

### Objective

Build a thin wrapper that:

1. loads Qwen 2.5 1.5B through `transformers`;
2. runs normal forward inference;
3. extracts hidden states;
4. passes those hidden states into the existing anchor subsystems.

### Proposed file

- `src/model/qwen_anchor_overlay.py`

### Expected components

- Qwen loading
- hidden-state extraction
- adapter from Qwen hidden states to current anchor interfaces
- anchor diagnostics output:
  - active anchors
  - contradiction pressure
  - viability
  - dead-end count
  - proposal candidate statistics

### Explicit non-goals

At this stage, do **not**:

- modify Qwen attention blocks
- retrain Qwen
- inject proposal logits back into generation
- add LoRA

## Phase 2 — Probe harness on Qwen

### Objective

Run the existing semantic probe families on top of the Qwen overlay.

### Proposed file

- `scripts/run_qwen_anchor_probe.py`

### Probe families

Start with existing toy families:

- quantifier
- proof mode
- induction
- formal limit

Then, if the overlay looks promising, extend to:

- code/instruction conflicts
- API/framework conflicts
- long-context instruction retention

### Success criterion

We want to see whether stable/conflict separation becomes cleaner on a strong pretrained model than on the current tiny backbone.

## Phase 3 — Compare ABPT vs Qwen overlay

### Objective

Produce a direct comparison between:

- `ABPTAnchorV1`
- Qwen + anchor overlay

### Proposed outputs

- `docs/research/qwen_anchor_probe_report.md`
- optionally a compare report script

### Main metrics

- contradiction pressure
- viability
- dead-end count
- stable/conflict separation
- proposal candidate frequency
- proposal score distribution

### Interpretation target

This phase should answer one important question:

> was the small custom backbone the main bottleneck, or is the anchor idea itself weak?

## Phase 4 — Better anchor scoring proxies

Only after the basic Qwen overlay is stable.

### Objective

Test scoring methods that sit between:

- cheap local hidden-state delta
- expensive leave-one-out KL

The first candidate is a future-conditioned gradient scorer:

- compute a short autoregressive future loss window;
- backpropagate that loss to token positions or hidden states;
- treat high-gradient positions as candidates for semantically important spans.

### Why this matters

If local delta is too myopic and full KL is too expensive, then future-conditioned attribution may be a practical middle ground.

### Current status

An exploratory future-gradient probe exists, but results are still mixed. The method looks more promising as a position-specific anchor relevance signal than as a single prompt-level scalar.

## Phase 5 — Minimal intervention mode

Only attempt this if Phases 1–3 look meaningful.

### Objective

Let the anchor engine influence generation in a minimal way, without rewriting Qwen.

### Candidate mechanisms

- light logit reranking
- additive bias from anchor state
- soft preference for proposal-consistent continuations

### Constraint

Keep intervention small and measurable. Avoid building a full alternate decoding stack too early.

## Phase 6 — Lightweight trainable heads

Only if earlier phases are promising.

### Objective

Train only small anchor-specific components on top of a mostly frozen Qwen backbone.

### Candidate trainable parts

- anchor scoring head
- proposal head
- gate calibration head

### Non-goal

Do not full-finetune the whole 1.5B model as a first move.

## Architectural sketch

```text
input_ids
  -> Qwen forward
  -> hidden_states
  -> Anchor Overlay
  -> diagnostics
  -> optional small generation bias
```

This keeps the integration modular and falsifiable.

## Risks

1. Qwen hidden states may still not produce clean anchor signals.
2. Current anchor heuristics may be too tied to the small model regime.
3. Proposal routing may remain weak even on a strong backbone.
4. Overlay diagnostics may work while intervention still fails.

All of these outcomes are informative.

## Practical implementation order

1. `src/model/qwen_anchor_overlay.py`
2. `scripts/run_qwen_anchor_probe.py`
3. Qwen-vs-ABPT compare report
4. future-influence / attribution-style scoring experiments
5. only then minimal intervention during generation
6. only then lightweight trainable heads

## Short conclusion

The next serious test of the idea is not a larger custom ABPT training run.
It is an anchor overlay on top of a stronger pretrained model such as Qwen 2.5 1.5B.
