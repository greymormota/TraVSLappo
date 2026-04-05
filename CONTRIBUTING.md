# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install SUMO separately and ensure `sumo`, `sumolib`, and `traci` are available in your environment.

## Branching

- `main`: stable branch for releasable changes
- feature branches: `feature/<topic>`
- bugfix branches: `fix/<topic>`

## Pull requests

1. Keep changes focused and small.
2. Update docs when behavior or commands change.
3. Include the exact command used for training or evaluation.
4. Avoid committing generated results, model checkpoints, or large binaries.

## Recommended checks

```bash
python -m compileall algorithms envs runner train utils
python train/train.py --help
```
