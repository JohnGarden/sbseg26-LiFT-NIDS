"""Owns the experiment results schema.

The Axis 1 / Axis 2 runners each hand-built their results.json record, stamped
provenance inline, and flattened a *different* subset of fields into the shared
registry. This module gives that schema one home:

* :func:`provenance` — provenance stamping;
* :func:`registry_row` — the one rule for flattening a record to scalar
  registry columns;
* :func:`write_result` — the save-to-JSON + append-to-registry seam;
* :func:`build_axis1_record` / :func:`build_axis2_record` — the canonical
  per-axis result records.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from lift_nids.utils.io import append_results_registry, save_results


def provenance() -> dict[str, str]:
    """Return ``{"timestamp"}`` for stamping a result record.

    The timestamp is the local wall-clock time to second resolution, matching
    the ISO-8601 format the runners have always written.
    """
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def registry_row(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a result record to scalar columns for the append-only registry.

    Drops nested ``dict``/``list`` values (metric blocks, per-seed lists,
    trajectories) and keeps scalar provenance/config fields. One rule for every
    runner, so registry rows are consistent across axes.
    """
    return {k: v for k, v in record.items() if not isinstance(v, (dict, list))}


def write_result(
    record: dict[str, Any],
    results_path: str | Path,
    *,
    registry_dir: str | Path | None = None,
) -> None:
    """Persist a result record and, optionally, register its scalar row.

    Saves the full ``record`` to ``results_path`` as JSON, then (when
    ``registry_dir`` is given) appends :func:`registry_row` of it to the
    append-only experiment registry. Collapses the save + register pattern the
    runners repeated.
    """
    save_results(record, Path(results_path))
    if registry_dir is not None:
        append_results_registry(registry_row(record), Path(registry_dir))


def build_axis1_record(
    *,
    model_family: str,
    seeds: list[int],
    n_trials: int,
    num_workers: int,
    effective_batch_size: int,
    best_hpo_val_f1: float,
    best_params: dict[str, Any],
    per_seed_metrics: list[dict[str, Any]],
    metrics: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Assemble the CICIoT2023 (Axis 1) per-model results record.

    ``dataset``/``axis`` are fixed for this axis; provenance is stamped last.
    """
    return {
        "model_family": model_family,
        "dataset": "CICIoT2023",
        "axis": 1,
        "seeds": seeds,
        "n_trials": n_trials,
        "num_workers": num_workers,
        "effective_batch_size": effective_batch_size,
        "best_hpo_val_f1": best_hpo_val_f1,
        "best_params": best_params,
        "per_seed_metrics": per_seed_metrics,
        "metrics": metrics,
        "elapsed_seconds": round(elapsed_seconds, 1),
        **provenance(),
    }


def build_axis2_record(
    *,
    model_family: str,
    seeds: list[int],
    horizon: int | str,
    years: list[int],
    n_trials: int,
    sample_frac: float | None,
    run_mode: str,
    best_hpo_val_f1: float,
    best_params: dict[str, Any],
    effective_batch_size: int,
    per_seed_results: list[dict[str, Any]],
    metrics: dict[str, Any],
    efficiency_summary: dict[str, Any],
    save_checkpoints: bool,
    save_attention_artifacts: bool,
    attention_samples_per_group: int | None,
    artifacts_dir: str | None,
    num_workers: int,
    xgboost_device: str,
    hpo_device: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Assemble the MAWIFlow (Axis 2) per-horizon results record.

    ``dataset``/``axis`` and the single-anchor-year HPO-policy metadata are fixed
    for this axis; ``hpo_anchor_year`` is derived from the first training year;
    provenance is stamped last.
    """
    return {
        "model_family": model_family,
        "dataset": "MAWIFlow",
        "axis": 2,
        "seeds": seeds,
        "horizon": horizon,
        "years": years,
        "n_trials": n_trials,
        "sample_frac": sample_frac,
        "run_mode": run_mode,
        "best_hpo_val_f1": best_hpo_val_f1,
        "best_params": best_params,
        "effective_batch_size": effective_batch_size,
        # HPO policy metadata — single-anchor-year strategy (see README)
        "hpo_policy": "single_anchor_year",
        "hpo_anchor_year": int(years[0]),
        "hpo_split": "temporal_60_20_20",
        "hpo_reused_across_windows": True,
        "per_seed_results": per_seed_results,
        "metrics": metrics,
        "efficiency_summary": efficiency_summary,
        # Interpretability artifact metadata
        "save_checkpoints": save_checkpoints,
        "save_attention_artifacts": save_attention_artifacts,
        "attention_samples_per_group": attention_samples_per_group,
        "artifacts_dir": artifacts_dir,
        # Execution metadata
        "num_workers": num_workers,
        "xgboost_device": xgboost_device,
        "hpo_device": hpo_device,
        "elapsed_seconds": round(elapsed_seconds, 1),
        **provenance(),
    }
