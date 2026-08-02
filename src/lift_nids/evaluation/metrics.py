"""Standard classification metrics for both evaluation axes."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute F1 macro, F1 per-class, Precision, Recall, AUC-ROC, AUC-PR.

    Args:
        y_true: Ground-truth binary labels (N,).
        y_pred: Predicted binary labels (N,).
        y_proba: Predicted probabilities for the positive class (N,).
        threshold: Decision threshold (informational only — y_pred is already thresholded).

    Returns:
        Dict with keys: f1_macro, f1_0, f1_1, precision, recall, auc_roc, auc_pr.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    result: dict[str, float] = {
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_0":      float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_1":      float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float("nan"),
        "auc_pr":    float("nan"),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba).ravel()
        try:
            result["auc_roc"] = float(roc_auc_score(y_true, y_proba))
            result["auc_pr"]  = float(average_precision_score(y_true, y_proba))
        except ValueError:
            pass  # e.g. only one class present in y_true

    return result


def f1_plus(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 of the positive (attack) class — used for temporal trajectory."""
    return float(
        f1_score(
            np.asarray(y_true).ravel(),
            np.asarray(y_pred).ravel(),
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )


def _extract_f1_plus(metrics: dict) -> float:
    """Return F1 of the positive class from a compute_classification_metrics result.

    Always reads 'f1_1'; never falls back to 'f1_macro' so that F1+ records
    and nAUT values are not silently contaminated by the macro average.
    """
    return float(metrics.get("f1_1", 0.0))
