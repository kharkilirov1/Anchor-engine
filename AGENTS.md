# ABPT — Adaptive Branching Plastic Transformer with Attention Residuals

## Project Type
Research prototype — small language model architecture experiment.

## Language
All communication in Russian. Code, comments, variable names in English.

## Architecture
MVP Stage A: transformer backbone + AttnRes + 2 branch heads + verifier + plastic layer.
See `docs/superpowers/plans/` for implementation plan.
See `docs/context.md` for full architecture context from the research proposal.

## Code Rules
- PyTorch only, no custom CUDA kernels in MVP
- All modules togglable via config (for ablation)
- Type hints on all function signatures
- No docstrings unless logic is non-obvious
- Tests with pytest, test each module in isolation
- Config via dataclasses, not yaml/json (single source of truth)
- Target: Python 3.10+, PyTorch 2.x, works on CPU and CUDA

## File Organization
- `src/model/` — all model components (one file per module)
- `src/data/` — data loading and synthetic tasks
- `src/utils/` — metrics, logging, helpers
- `configs/` — ablation config presets
- `tests/` — pytest tests mirroring src/ structure
- `train.py` — training entry point
- `evaluate.py` — evaluation entry point

## Conventions
- Model forward() returns dict with all intermediate values (for metrics/debugging)
- Loss functions return dict {total_loss, component_losses}
- All tensor shapes documented in comments where non-obvious
- Use einops for complex reshapes

## Testing
- Each module has unit test verifying forward pass shape correctness
- Integration test: full model forward + backward on random data
- Ablation test: each config produces valid model

## Commit Style
- feat: new module/feature
- fix: bug fix
- test: test additions
- docs: documentation
- refactor: code restructuring
