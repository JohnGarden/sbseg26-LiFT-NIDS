"""Experiment metadata helpers."""

from __future__ import annotations

import re
from pathlib import Path

EXPERIMENT_NAME_PATTERN = re.compile(
    r"^exp_\d{3}_[a-z0-9]+_(static|temporal|crossdataset)_[a-z0-9_]+$"
)


def is_valid_experiment_name(name: str) -> bool:
    """Validate experiment naming convention used by LiFT-NIDS."""
    return bool(EXPERIMENT_NAME_PATTERN.fullmatch(name))


def infer_experiment_name_from_path(path: str | Path) -> str:
    """Infer experiment name from a config filename."""
    return Path(path).stem
