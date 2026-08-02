"""Evaluation subpackage: metrics, temporal metrics, statistical tests, CD diagram."""

from lift_nids.evaluation.cd_diagram import plot_cd_diagram
from lift_nids.evaluation.metrics import compute_classification_metrics, f1_plus
from lift_nids.evaluation.statistical_tests import (
    bootstrap_ci,
    friedman_nemenyi,
    wilcoxon_test,
)
from lift_nids.evaluation.temporal_metrics import (
    compute_naut,
    f1_plus_trajectory,
    median_trajectory,
)

__all__ = [
    "compute_classification_metrics",
    "f1_plus",
    "f1_plus_trajectory",
    "compute_naut",
    "bootstrap_ci",
    "wilcoxon_test",
    "friedman_nemenyi",
    "median_trajectory",
    "plot_cd_diagram",
]
