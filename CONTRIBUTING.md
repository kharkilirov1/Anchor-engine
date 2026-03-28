# Contributing to ABPT

Thanks for taking this repo seriously enough to inspect it.

## Ground rules

This repository is an exploratory research codebase, not a polished framework. Please optimize for clarity, reproducibility, and honest reporting rather than for hype.

Good contributions:
- reproducible bug reports
- small targeted fixes
- clearer experiment logging
- checkpoint loading / transfer utilities
- cleaner dataset pipelines
- stronger tests
- negative results with exact commands and settings
- critique of the theory when grounded in code or experiments

Less useful contributions:
- broad architectural rewrites without evidence
- benchmark claims without commands, configs, and artifacts
- cosmetic refactors that make experiment history harder to read

## Development setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
pytest -q
```

## Project conventions

Please follow `AGENTS.md`.

Highlights:
- communication in Russian
- code/comments/identifiers in English
- PyTorch only for model code
- config via dataclasses
- tests with pytest
- minimal, architecture-aware changes
- no loud claims in docs

## Pull request expectations

A useful PR should usually include:
1. what changed
2. why it changed
3. how it was verified
4. what tradeoff or limitation remains

If you change model behavior, include at least one of:
- a unit test
- an integration test
- a short report artifact
- a command sequence to reproduce the result

## Research etiquette

If your conclusion is that a subsystem is weak, brittle, or unnecessary, that is a valid contribution. This repository is open precisely so that weak ideas can be falsified cleanly.
