PYTHON ?= 3.11
UV ?= uv

.PHONY: bootstrap sync lint format test smoke validate run

bootstrap:
	$(UV) python install $(PYTHON)
	$(UV) sync --all-groups

sync:
	$(UV) sync --all-groups

lint:
	$(UV) run ruff check src/ scripts/ tests/

format:
	$(UV) run ruff check --fix src/ scripts/ tests/

test:
	$(UV) run pytest

smoke:
	$(UV) run python scripts/validate_config.py --config configs/experiments/exp_001_ciciot_static_xgb.yaml
	$(UV) run python scripts/run_experiment.py --config configs/experiments/exp_001_ciciot_static_xgb.yaml

validate:
	$(UV) run python scripts/validate_config.py --config configs/experiments/exp_001_ciciot_static_xgb.yaml

run:
	$(UV) run python scripts/run_experiment.py --config configs/experiments/exp_002_ciciot_static_mlp.yaml
