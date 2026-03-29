# Current Status

Date: 2026-03-29

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

A direct proxy comparison currently suggests:

- delta-pressure wins in `5/8` families;
- delta-viability wins in `5/8` families;
- raw mean future influence wins in only `3/8` families;
- anchor-position future influence wins in `5/8` families.

So the current best reading is that future-conditioned attribution is more useful as an anchor-local relevance probe than as a global prompt-level scorer.

The next refinement step is now partially implemented:

- high future-influence positions can be grouped into contiguous spans;
- those spans can be compared against current active anchor spans to measure overlap and possible detector misses.

On the current 8-family Qwen suite, the span miss analysis splits the families into four buckets:

- aligned: `proof_mode`, `entity_property`
- future_rescue: `api_framework`, `quantifier`
- delta_only: `instruction_constraints`, `legal_scope`
- both_weak: `induction`, `units`

This is useful because it suggests future-conditioned attribution may already help in families where the current detector underperforms, even though it is still weaker as a global scorer.

There is now also a lightweight proposal-hint extraction path:

- non-overlapping high future-influence spans can be surfaced as candidate proposal hints;
- this is especially relevant for the current future-rescue families (`api_framework`, `quantifier`);
- these hints are still diagnostic and exploratory, not yet wired into decoding or revision.

The new offline auxiliary-proposal evaluation is slightly stronger than the earlier raw-hint reports:

- conflict auxiliary proposal-count wins in `5/8` families;
- conflict auxiliary proposal-score wins in `5/8` families;
- mean conflict-minus-stable auxiliary proposal-count gap is `+0.6250`;
- mean conflict-minus-stable auxiliary proposal-score gap is `+0.0352`.

This is still not enough to claim a working proposal system. However, it is enough to say that the future-influence path is beginning to produce proposal-like conflict signals on top of a real pretrained model.

The most useful current rescue examples remain:

- `api_framework_conflict` → `a synchronous Django view`
- `quantifier_conflict` → `a witness`

Those spans are now visible as auxiliary proposal-like candidates even when the base detector underperforms on the same family.

An additional offline revision simulation is now in place on top of those auxiliary proposals. This is still heuristic and should not be read as a solved proposal mechanism, but it does produce non-trivial movement:

- conflict matched-anchor wins in `3/8` families;
- conflict revise-gain wins in `3/8` families;
- mean conflict-minus-stable revise-gain gap is `+0.2500`;
- mean conflict-minus-stable retire-delta gap is `-0.1250`.

The auxiliary matcher is now one-to-one between hint spans and anchors. That reduced the earlier overmatching problem on some stable prompts, especially `quantifier_stable`.

The most promising current family is `api_framework`:

- stable matched anchors: `1`
- conflict matched anchors: `2`
- stable revise gain: `0`
- conflict revise gain: `2`

That is the first useful sign that future-hint proposals can begin to push the revision controller toward `revise` in a family where the base detector was previously weak.

The weakest current rescue family is still `quantifier`. There, the auxiliary path now behaves more symmetrically:

- stable matched anchors: `1`
- conflict matched anchors: `1`
- stable revise gain: `1`
- conflict revise gain: `1`

So the current bottleneck has shifted from pure overmatching to **arbiter calibration**. The hint `a witness` is present, but the revision controller still does not treat it as stronger than the stable control.
