PYTHON ?= python
PIP ?= pip

.PHONY: install dev check train test clean

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -e .

check:
	$(PYTHON) -m compileall algorithms envs runner train utils
	$(PYTHON) train/train.py --help

train:
	$(PYTHON) train/train.py

test:
	$(PYTHON) test.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
