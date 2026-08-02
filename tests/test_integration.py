"""End-to-end integration tests using synthetic data (no Parquet required).

These tests verify that the full training pipeline works for each model family:
train → Optuna search → best_params returned, best_value is a valid float.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lift_nids.data.dataloader import build_dataloader
from lift_nids.data.dataset import FlowDataset, LazySequenceFlowDataset
from lift_nids.evaluation.metrics import compute_classification_metrics
from lift_nids.training.optuna_search import run_optuna_search

RNG = np.random.default_rng(0)
N = 200
F = 10
W = 8
N_CLASSES = 2
SPLIT = int(0.8 * N)


def _tabular_arrays():
    X = RNG.standard_normal((N, F)).astype(np.float32)
    y = RNG.integers(0, N_CLASSES, size=N)
    y[:N_CLASSES] = np.arange(N_CLASSES)  # guarantee both classes
    return (X[:SPLIT], y[:SPLIT]), (X[SPLIT:], y[SPLIT:])


def _tabular_loaders():
    (X_tr, y_tr), (X_val, y_val) = _tabular_arrays()
    return (
        build_dataloader(FlowDataset(X_tr, y_tr), batch_size=32, shuffle=True),
        build_dataloader(FlowDataset(X_val, y_val), batch_size=32),
    )


def _seq_loaders():
    X = RNG.standard_normal((N, F)).astype(np.float32)  # base flows; windows built lazily
    y = RNG.integers(0, N_CLASSES, size=N)
    y[:N_CLASSES] = np.arange(N_CLASSES)
    return (
        build_dataloader(
            LazySequenceFlowDataset(X[:SPLIT], y[:SPLIT], W, stride=1),
            batch_size=16, shuffle=True,
        ),
        build_dataloader(
            LazySequenceFlowDataset(X[SPLIT:], y[SPLIT:], W, stride=1),
            batch_size=16,
        ),
    )


def _is_valid_score(v: object) -> bool:
    return isinstance(v, float) and not math.isnan(v) and 0.0 <= v <= 1.0


class TestE2EPipeline:
    def test_e2e_xgboost(self) -> None:
        (X_tr, y_tr), (X_val, y_val) = _tabular_arrays()
        result = run_optuna_search(
            model_type="xgboost",
            train_data=(X_tr, y_tr),
            val_data=(X_val, y_val),
            n_features=F,
            n_classes=N_CLASSES,
            n_trials=1,
            max_epochs=1,
            device="cpu",
        )
        assert "best_params" in result
        assert result["n_trials_completed"] >= 1
        assert _is_valid_score(result["best_value"])

    def test_e2e_mlp(self) -> None:
        train_dl, val_dl = _tabular_loaders()
        result = run_optuna_search(
            model_type="mlp",
            train_data=train_dl,
            val_data=val_dl,
            n_features=F,
            n_classes=N_CLASSES,
            n_trials=1,
            max_epochs=2,
            device="cpu",
        )
        assert "best_params" in result
        assert _is_valid_score(result["best_value"])

    def test_e2e_cnn_bilstm(self) -> None:
        train_dl, val_dl = _seq_loaders()
        result = run_optuna_search(
            model_type="cnn_bilstm",
            train_data=train_dl,
            val_data=val_dl,
            n_features=F,
            n_classes=N_CLASSES,
            n_trials=1,
            max_epochs=2,
            device="cpu",
        )
        assert "best_params" in result
        assert _is_valid_score(result["best_value"])

    def test_e2e_transformer(self) -> None:
        train_dl, val_dl = _seq_loaders()
        result = run_optuna_search(
            model_type="transformer",
            train_data=train_dl,
            val_data=val_dl,
            n_features=F,
            n_classes=N_CLASSES,
            n_trials=1,
            max_epochs=2,
            device="cpu",
        )
        assert "best_params" in result
        assert _is_valid_score(result["best_value"])

    def test_compute_metrics_valid_output(self) -> None:
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.integers(0, 2, size=100)
        y_score = rng.uniform(0, 1, size=100).astype(np.float32)
        m = compute_classification_metrics(y_true, y_pred, y_score)
        assert _is_valid_score(m["f1_macro"])
        assert _is_valid_score(m["auc_roc"])
        assert _is_valid_score(m["auc_pr"])

    def test_unknown_model_type_raises(self) -> None:
        (X_tr, y_tr), (X_val, y_val) = _tabular_arrays()
        with pytest.raises(ValueError, match="Unknown model_type"):
            run_optuna_search(
                model_type="invalid_model",
                train_data=(X_tr, y_tr),
                val_data=(X_val, y_val),
                n_features=F,
                n_classes=N_CLASSES,
                n_trials=1,
            )
