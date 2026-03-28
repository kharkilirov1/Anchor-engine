# ABPT

Adaptive Branching Plastic Transformer (ABPT) is an open research repository for small language-model experiments around two ideas:

1. a compact modular transformer stack (`AttnRes`, branches, verifier, plasticity), and
2. a newer anchor-centric line of work (`ABPTAnchorV1`) that treats some spans as long-horizon semantic anchors whose failure can destabilize future generation.

This repository is **not** presented as a finished architecture or a validated breakthrough. It is a working prototype with code, tests, notes, and experiment utilities kept in one place so other people can inspect, reproduce, criticize, or extend it.

## Current status

What exists today:
- a trainable baseline model stack in `src/model/abpt.py`
- an anchor-centric prototype in `src/model/abpt_anchor_v1.py`
- synthetic probe/report tooling for anchor behavior
- training/evaluation entry points with history logging
- self-contained Colab notebook for baseline-vs-anchor comparison
- a passing pytest suite

What is **not** established today:
- that the anchor formulation is fundamentally better than standard modeling
- that proposal/revision machinery works under realistic training pressure
- that the current synthetic tasks are sufficient evidence of the theory
- that the current code is ready for production or large-scale research use without cleanup

## Recent empirical picture

The most recent TinyStories BPE compare run (external Colab run, not a polished benchmark suite) looked like this:

- baseline final `val_bpb`: `6.2377`
- anchor final `val_bpb`: `5.9606`
- anchor run also improved internal anchor health metrics:
  - `anchor_contradiction`: `0.8429 -> 0.7817`
  - `anchor_viability`: `0.1119 -> 0.2319`
  - `anchor_dead_end`: `75 -> 53`
- but proposal routing still stayed inactive:
  - `proposal_influence = 0`
  - `proposal_blend = 0`

Interpretation: the current anchor prototype is not obviously useless, but its strongest proposed mechanism (proposal/revision path) is still not validated.

## Repository map

### Core code
- `src/model/backbone.py` — shared transformer backbone
- `src/model/abpt.py` — baseline/stage-A integrated model
- `src/model/abpt_anchor_v1.py` — anchor-centric prototype
- `src/model/anchor_*.py` — anchor subsystems
- `src/data/` — Shakespeare, synthetic, Stack/TinyStories BPE loaders
- `src/utils/` — metrics and helpers

### Entrypoints
- `train.py`
- `evaluate.py`

### Reports / theory
- `docs/research/PROJECT_MAP.md`
- `docs/research/ANCHOR_THEORY.md`
- `docs/research/ARCHITECTURE_V1.md`
- `docs/research/CURRENT_STATUS.md`
- `program.md`

### Notebook
- `notebooks/colab_anchor_v1_upload_only.ipynb`
  - upload-only notebook
  - restores a project snapshot automatically
  - runs baseline vs anchor on TinyStories BPE
  - prints a compare report at the end

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Run tests

```bash
pytest -q
```

### 3. Small local smoke run

Baseline:

```bash
python train.py --preset toy --stage a --device cpu --steps 1
```

Anchor:

```bash
python train.py --preset toy --stage anchor --device cpu --steps 1
```

### 4. Evaluation

```bash
python evaluate.py --preset toy --stage anchor --device cpu --batches 1
```

## Datasets currently supported

- `shakespeare`
- `anchor-synthetic`
- `the-stack`
- `the-stack-bpe`
- `tinystories-bpe`

Example:

```bash
python train.py --preset toy --stage anchor --dataset tinystories-bpe --device cuda --steps 5000
```

## Open-source framing

This repo is being opened as a research artifact, not as a finished claim. The intended value is:
- inspectable code
- reproducible small experiments
- theory-to-code traceability
- room for external criticism

If the anchor idea turns out weak, that is still useful information. If it turns out partially useful, this repo should make that visible without inflated claims.

## Known limitations

- proposal/revision path is not yet convincingly trained
- some docs reflect earlier phases of the project and remain for provenance
- dataset pipelines are pragmatic, not final benchmark pipelines
- checkpoint transfer / pretrained-backbone initialization is not yet the main workflow
- results are still small-scale and exploratory

## If you want to contribute

Please read:
- `CONTRIBUTING.md`
- `AGENTS.md`

The most useful contributions right now are usually:
- reproductions
- bug reports
- cleaner checkpoint/init workflows
- better conflict-supervised tasks
- negative results with clear methodology

## License

Apache-2.0 — see `LICENSE`.
