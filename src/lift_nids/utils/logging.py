"""Structured logging setup for LiFT-NIDS experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger with optional file handler.

    Args:
        name: Logger name (typically __name__ of the caller).
        log_file: If provided, also write to this file.
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured in this process

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_experiment_result(
    logger: logging.Logger,
    experiment_id: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
) -> None:
    """Log a structured experiment result entry."""
    logger.info("Experiment: %s", experiment_id)
    for k, v in metrics.items():
        logger.info("  %-20s = %s", k, f"{v:.4f}" if isinstance(v, float) else v)
    if params:
        logger.info("  params: %s", params)
