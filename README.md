# TraVSLappo

A GitHub repository project for **Multi-Agent Proximal Policy Optimization-based Variable Speed Limits control** in **SUMO traffic simulation**.

## What is included

- MAPPO core implementation in `algorithms/`
- environment wrappers and SUMO environment code in `envs/`
- training runners in `runner/`
- training and analysis scripts in `train/`
- utility modules in `utils/`
- SUMO network and route assets in `sumo/`
- helper scripts in `scripts/`
- archived legacy code in `legacy/`

## Repository layout

```text
.
├── algorithms/
├── envs/
├── runner/
├── scripts/
├── sumo/
├── train/
├── utils/
├── docs/
├── artifacts/
├── legacy/
├── config.py
├── test.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- SUMO installed locally
- Python packages from `requirements.txt`

## Quick start

```bash
git clone https://github.com/greymormota/TraVSLappo.git
cd TraVSLappo
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or use:

```bash
bash scripts/setup_env.sh
```

## Training

Default training entrypoint:

```bash
python train/train.py
```

Example:

```bash
python train/train.py --algorithm_name mappo --experiment_name baseline --scenario_name MyEnv --num_agents 12
```

## Evaluating

### Supported controllers
The evaluation script supports the following controllers:
- `mappo`: trained MAPPO policy 
- `rule_based`: occupancy-threshold rule-based baseline
- `no_control`: no active control baseline

### Supported demand scenarios
The following demand levels are supported:

- `undersaturated` (free flow)
- `saturated` (capicity flow)
- `oversaturated` (Extreme, per)

You can evaluate all demand settings together or specify only a subset. :contentReference[oaicite:3]{index=3}

### Basic usage

Evaluate all controllers under all demand scenarios:

```bash
python train/eval_training_matrix.py \
  --model_dir ./models \
  --controllers all \
  --demands all \
  --out_dir ./results_eval/training_matrix
```

## Notes

- The main training pipeline starts from `train/train.py`.
- --model_dir is required when evaluating mappo
- evaluation horizon defaults to --eval_episode_length, and can be overridden with --episode_length_eval

## License

MIT. See `LICENSE`.
