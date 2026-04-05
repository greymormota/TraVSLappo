# Project Structure

- `algorithms/`: MAPPO policy, actor-critic, and algorithm utilities
- `envs/`: environment wrappers and SUMO-facing environment implementation
- `runner/`: shared and separated runner logic
- `sumo/`: SUMO network and route configuration files
- `train/`: training and plotting/evaluation scripts
- `utils/`: replay buffer, normalization, and helper utilities
- `scripts/`: shell entrypoints preserved from the original project
- `legacy/`: old or non-primary scripts kept for reference only
- `docs/`: project documentation
- `artifacts/`: notes about excluded generated assets

## Primary entrypoints

- Training: `python train/train.py`

## Notes

