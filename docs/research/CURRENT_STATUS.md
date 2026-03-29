# Current Status

Date: 2026-03-28

## What this repository is right now

ABPT is currently best understood as a compact open research repository, not a mature model release.

The repository contains:
- an earlier modular LM architecture (`AttnRes`, branches, verifier, plasticity)
- a newer anchor-centric prototype (`ABPTAnchorV1`)
- probe/report tooling for anchor behavior
- dataset loaders for small local and Colab-friendly experiments
- an upload-only Colab notebook for baseline-vs-anchor comparison

Generated checkpoints, caches, and ad-hoc run artifacts are intentionally excluded from the public repository snapshot.

## What has some empirical support

A recent TinyStories BPE compare run showed:
- baseline final `val_bpb = 6.2377`
- anchor final `val_bpb = 5.9606`

Within the anchor run:
- `anchor_contradiction` decreased
- `anchor_viability` increased
- `anchor_dead_end` decreased

This is enough to say the current anchor code is not obviously inert.

## What remains unresolved

The main unresolved issue is the proposal/revision path.

In the latest compare run:
- `proposal_influence = 0`
- `proposal_blend = 0`

So the strongest intended mechanism of the anchor framing is still not convincingly active in training.

## Current honest interpretation

A fair reading of the repo at this point is:
- the baseline is trainable
- the anchor overlay is also trainable
- the anchor version can improve some metrics in small runs
- the proposal/revision story is still weak or undertrained
- the project is worth inspection, but not worth inflated claims

## Most reasonable next technical step

If development continues, the cleanest next move is likely:
1. save a trained baseline checkpoint
2. initialize `ABPTAnchorV1` from the pretrained backbone
3. train anchor-specific modules on top of an already learned world model

That path is more consistent with the project’s current evidence than training the full anchor stack from scratch.

In parallel, the current approved external-test direction is:

1. attach the anchor engine as an overlay to `Qwen2.5-1.5B`;
2. run the existing semantic probes on Qwen hidden states;
3. compare whether anchor diagnostics become more meaningful on a stronger pretrained backbone.

## Qwen overlay update

The Qwen overlay path is now live enough to produce reproducible diagnostics on a real pretrained model.

Current status on `Qwen/Qwen2.5-1.5B`:

- anchor probe suite: `16` paired stable/conflict cases across `8` families
- threshold calibration: implemented
- stable/conflict separation: present but still modest
- proposal path: still inactive

An additional experimental scorer was added as a midpoint between local hidden-state delta and full leave-one-out KL:

- future-gradient influence on a short autoregressive loss window

This scorer does **not** yet cleanly separate stable vs conflict cases by simple prompt-level mean. However, it does show stronger signal at active anchor positions in several conflict prompts. The current honest interpretation is:

- the future-influence idea is worth keeping as an exploratory direction;
- it is not yet a replacement for the existing anchor diagnostics;
- the most useful signal may be position-specific span relevance rather than one scalar per prompt.
