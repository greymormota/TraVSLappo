#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

echo "Environment ready. Activate it with: source .venv/bin/activate"
