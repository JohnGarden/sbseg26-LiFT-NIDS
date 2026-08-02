#!/usr/bin/env python3
"""CLI to validate LiFT-NIDS experiment configuration naming."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from lift_nids.utils import infer_experiment_name_from_path, is_valid_experiment_name


def main() -> None:
    parser = ArgumentParser(description="Validate LiFT-NIDS experiment config path.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    experiment_name = infer_experiment_name_from_path(config_path)
    if not is_valid_experiment_name(experiment_name):
        raise SystemExit(
            "Invalid experiment name. Expected format: "
            "exp_<id>_<dataset>_<split_protocol>_<model>."
        )

    print(f"[ok] Config naming validated: {config_path}")


if __name__ == "__main__":
    main()
