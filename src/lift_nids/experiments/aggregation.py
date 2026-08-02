"""Aggregate per-seed metrics into mean / std / bootstrap-CI summaries.

The Axis 1 and Axis 2 runners each carried an identical ``_aggregate_metrics``
that differed only in which metric keys it iterated. This is the single home for
that logic: pass the keys in.
"""

from __future__ import annotations

import numpy as np

from lift_nids.evaluation.statistical_tests import bootstrap_ci
from lift_nids.experiments.constants import N_BOOTSTRAP


def aggregate_metrics(
    per_seed: list[dict],
    metric_keys: list[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute mean, std, and bootstrap 95% CI for each metric across seeds.

    Args:
        per_seed: One dict of scalar metrics per repetition (seed). Missing keys
            and NaNs are ignored per metric.
        metric_keys: Metrics to aggregate; also fixes the output ordering.
        n_bootstrap: Bootstrap resamples for the CI.
        seed: Seed for the bootstrap resampling (fixed for reproducibility).

    Returns:
        ``{metric: {"mean", "std", "ci_lower", "ci_upper"}}``. A metric with no
        valid (non-NaN) values maps to all-NaN. ``std`` uses ``ddof=1`` and is
        ``0.0`` for a single value.
    """
    result: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.array([r.get(key, float("nan")) for r in per_seed], dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            result[key] = {
                "mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"),
            }
            continue
        lo, hi = bootstrap_ci(valid, n_bootstrap=n_bootstrap, seed=seed)
        result[key] = {
            "mean":     float(np.mean(valid)),
            "std":      float(np.std(valid, ddof=1) if len(valid) > 1 else 0.0),
            "ci_lower": lo,
            "ci_upper": hi,
        }
    return result
