"""Shared experiment constants for the LiFT-NIDS evaluation axes.

Single source of truth for the seed set, bootstrap sample count, model roster,
and per-axis metric-key lists that the Axis 1 / Axis 2 runners and the
post-processing scripts all consume. Previously each script re-declared these,
so a change to (say) the seed set had to be made in four places.
"""

from __future__ import annotations

# R=5 repetitions with fixed, distinct seeds.
SEEDS: list[int] = [42, 123, 456, 789, 1024]

# Bootstrap resamples for the 95% CI on each aggregated metric.
N_BOOTSTRAP: int = 1000

# The four model families compared on both axes.
ALL_MODELS: list[str] = ["xgboost", "mlp", "cnn_bilstm", "transformer"]

# Metric keys aggregated per axis. Order defines the aggregated-dict order.
AXIS1_METRIC_KEYS: list[str] = [
    "f1_macro", "f1_0", "f1_1", "precision", "recall", "auc_roc", "auc_pr",
]
AXIS2_TEMPORAL_METRIC_KEYS: list[str] = ["nAUT_1", "nAUT_3", "nAUT_5"]
AXIS2_CLASS_METRIC_KEYS: list[str] = [
    "f1_macro", "f1_1", "precision", "recall", "auc_roc", "auc_pr",
]
