# Excluded artifacts

The original archive contained generated outputs and binary artifacts that are typically not committed to Git:

- `results/`
- `results_eval/`
- `models/`
- IDE metadata and Python cache files

Keep those directories local, or store large checkpoints with Git LFS if you need versioned model files.
