"""Tests for aggregate_metrics — per-seed mean/std/bootstrap-CI aggregation.

This is the single home for the aggregation both runners previously duplicated,
differing only in which metric keys they iterated.
"""

from __future__ import annotations

import numpy as np
import pytest

from lift_nids.experiments.aggregation import aggregate_metrics


def test_aggregate_metrics_mean_std_and_ci():
    per_seed = [{"f1_1": 0.90}, {"f1_1": 0.94}, {"f1_1": 0.92}]
    agg = aggregate_metrics(per_seed, ["f1_1"])

    assert set(agg.keys()) == {"f1_1"}
    m = agg["f1_1"]
    assert m["mean"] == pytest.approx(0.92)
    assert m["std"] == pytest.approx(float(np.std([0.90, 0.94, 0.92], ddof=1)))
    assert m["ci_lower"] <= m["mean"] <= m["ci_upper"]


def test_missing_and_nan_values_are_ignored_per_metric():
    per_seed = [{"f1_1": 0.9}, {}, {"f1_1": float("nan")}, {"f1_1": 0.8}]
    agg = aggregate_metrics(per_seed, ["f1_1"])
    # Only the two valid values (0.9, 0.8) contribute.
    assert agg["f1_1"]["mean"] == pytest.approx(0.85)


def test_metric_with_no_valid_values_is_all_nan():
    agg = aggregate_metrics([{"f1_1": float("nan")}, {}], ["auc_roc", "f1_1"])
    for key in ("auc_roc", "f1_1"):
        stats = agg[key]
        assert all(np.isnan(stats[s]) for s in ("mean", "std", "ci_lower", "ci_upper"))


def test_single_value_std_is_zero():
    agg = aggregate_metrics([{"f1_1": 0.77}], ["f1_1"])
    assert agg["f1_1"]["std"] == 0.0
    assert agg["f1_1"]["mean"] == pytest.approx(0.77)


def test_output_key_order_follows_metric_keys():
    keys = ["nAUT_1", "f1_macro", "recall"]
    agg = aggregate_metrics([{"nAUT_1": 0.5, "f1_macro": 0.6, "recall": 0.7}], keys)
    assert list(agg.keys()) == keys


def test_matches_legacy_inline_aggregation():
    """aggregate_metrics must reproduce the runners' old _aggregate_metrics."""
    from lift_nids.evaluation.statistical_tests import bootstrap_ci

    keys = ["f1_macro", "f1_0", "f1_1", "precision", "recall", "auc_roc", "auc_pr"]
    per_seed = [
        {k: 0.90 + 0.01 * i + 0.001 * j for j, k in enumerate(keys)}
        for i in range(5)
    ]

    def legacy(per_seed, metric_keys, n_bootstrap=1000):
        result = {}
        for key in metric_keys:
            values = np.array([r.get(key, float("nan")) for r in per_seed], dtype=float)
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                result[key] = {
                    "mean": float("nan"), "std": float("nan"),
                    "ci_lower": float("nan"), "ci_upper": float("nan"),
                }
                continue
            lo, hi = bootstrap_ci(valid, n_bootstrap=n_bootstrap, seed=42)
            result[key] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid, ddof=1) if len(valid) > 1 else 0.0),
                "ci_lower": lo,
                "ci_upper": hi,
            }
        return result

    assert aggregate_metrics(per_seed, keys) == legacy(per_seed, keys)
