"""Temporal evaluation metrics for the MAWIFlow forward-chaining axis.

Metrics follow the MAWIFlow benchmark (IM28):
  - F1+(y_train, y)    : F1 of the positive class at year y
  - F̃1+(Δt)            : Median F1+ trajectory over Δt = y - y_train steps
  - nAUT_H             : Normalized Area Under Time for horizon H ∈ {1, 3, 5}
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def f1_plus_trajectory(
    results: list[tuple[int, int, float]],
) -> dict[int, list[float]]:
    """Compute median F1+ trajectory F̃1+(Δt) from a list of run results.

    Args:
        results: List of (train_year, test_year, f1_plus) tuples across R repetitions
                 and all anchor years.

    Returns:
        Dict mapping Δt -> list of F1+ values (all values at that lag).
        Call np.median() on each list to get the median trajectory.
    """
    groups: dict[int, list[float]] = defaultdict(list)
    for train_year, test_year, fp in results:
        delta = int(test_year) - int(train_year)
        groups[delta].append(float(fp))
    return dict(groups)


def median_trajectory(
    results: list[tuple[int, int, float]],
) -> dict[int, float]:
    """Compute F̃1+(Δt) — median F1+ at each lag Δt.

    Convenience wrapper around f1_plus_trajectory that applies np.median.
    """
    raw = f1_plus_trajectory(results)
    return {delta: float(np.median(values)) for delta, values in raw.items()}


def compute_f1_trajectory(
    results_dict: dict[tuple[int, int], float | list[float]],
) -> dict[int, float]:
    """Compute F̃1+(Δt) from a results dictionary.

    Accepts experiment results keyed by (y_train, y_test) pairs and computes
    the median F1+ at each lag Δt = y_test − y_train.

    Args:
        results_dict: Maps ``(y_train, y_test)`` to a scalar F1+ value or a
            list of values (one per repetition, e.g. R=5 seeds).

    Returns:
        Dict mapping Δt -> median F1+.  Only Δt values with at least one
        observation are included.

    Example::

        results = {
            (2007, 2008): [0.82, 0.80, 0.83],
            (2008, 2009): 0.79,
        }
        traj = compute_f1_trajectory(results)
        # traj[1] == median across all Δt=1 entries
    """
    groups: dict[int, list[float]] = defaultdict(list)
    for (y_train, y_test), fp in results_dict.items():
        delta = int(y_test) - int(y_train)
        if isinstance(fp, (list, np.ndarray)):
            groups[delta].extend(float(v) for v in fp)
        else:
            groups[delta].append(float(fp))
    return {delta: float(np.median(vals)) for delta, vals in sorted(groups.items())}


def compute_naut(
    trajectory: dict[int, float],
    H: int,
    baseline: float = 0.0,
) -> float:
    """Compute normalized Area Under Time (nAUT_H) via trapezoidal rule.

    Implements the formula from IM28 (MAWIFlow):

        nAUT_H = (1/H) * Σ_{Δt=0}^{H-1} [ F̃1+(Δt) + F̃1+(Δt+1) ] / 2

    The sum integrates the F1+ trajectory over H intervals starting at Δt=0
    (the in-distribution internal test, same time window as training) up to
    Δt=H.  Missing lags default to 0.0.  The result lies in [0, 1].

    Δt=0 must be present in the trajectory for this formula to be meaningful.
    It comes from evaluating the model on a temporally held-out 20% slice of
    the training window (NOT on the training data itself).

    Args:
        trajectory: Dict mapping Δt -> median F1+ value (from median_trajectory).
            Must include Δt=0 for correct nAUT computation.
        H: Horizon in years (1, 3, or 5).
        baseline: Unused — kept for API compatibility.

    Returns:
        nAUT_H in [0, 1].

    Example::

        traj = {0: 0.9, 1: 0.7, 2: 0.5}
        compute_naut(traj, H=1)  # (0.9 + 0.7) / 2 = 0.8
        compute_naut(traj, H=2)  # ((0.9+0.7)/2 + (0.7+0.5)/2) / 2 = 0.7
    """
    total = sum(
        (trajectory.get(dt, 0.0) + trajectory.get(dt + 1, 0.0)) / 2
        for dt in range(H)
    )
    return float(total / H)
