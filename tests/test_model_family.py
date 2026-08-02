"""Tests for the ModelFamily adapters — one fit/predict/efficiency seam.

Each model family (tabular XGBoost, PyTorch tabular/sequential) is trained and
evaluated through the same interface, so callers stop branching on family at
every stage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lift_nids.training.family import (
    ModelFamily,
    Prediction,
    make_model_family,
)

N_FEATURES = 12
RNG = np.random.default_rng(0)

_TORCH_PARAMS = {
    "mlp": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_config_idx": 2,
        "dropout": 0.1,
        "batch_norm": True,
        "batch_size": 16,
    },
}

_XGB_PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.2}


def _tabular_arrays(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.standard_normal((n, N_FEATURES)).astype(np.float32)
    y = RNG.integers(0, 2, size=n).astype(np.int64)
    return X, y


def _loader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 16, *, shuffle: bool = False
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def test_xgboost_family_fit_predict_returns_prediction():
    X_tr, y_tr = _tabular_arrays()
    X_vl, y_vl = _tabular_arrays(32)
    X_te, y_te = _tabular_arrays(24)

    fam = make_model_family("xgboost", _XGB_PARAMS, N_FEATURES, seed=42, device="cpu")
    assert isinstance(fam, ModelFamily)

    fam.fit((X_tr, y_tr), (X_vl, y_vl))
    pred = fam.predict((X_te, y_te))

    assert isinstance(pred, Prediction)
    np.testing.assert_array_equal(pred.y_true, y_te)
    assert pred.y_pred.shape == (24,)
    assert pred.y_proba.shape == (24,)
    assert set(np.unique(pred.y_pred)).issubset({0, 1})


def test_xgboost_family_efficiency_record():
    X_tr, y_tr = _tabular_arrays()
    X_vl, y_vl = _tabular_arrays(32)
    X_te, y_te = _tabular_arrays(24)

    fam = make_model_family("xgboost", _XGB_PARAMS, N_FEATURES, seed=42, device="cpu")
    fam.fit((X_tr, y_tr), (X_vl, y_vl))
    fam.predict((X_te, y_te))
    eff = fam.efficiency(n_train_samples=len(X_tr), n_infer_samples=len(X_te))

    assert eff["device"] == "cpu"
    assert eff["peak_gpu_mem_bytes"] == 0
    assert eff["n_train_samples"] == len(X_tr)
    assert eff["n_infer_samples"] == len(X_te)
    assert eff["train_time_s"] >= 0.0


def test_make_model_family_dispatch_and_unknown():
    from lift_nids.training.family import TorchFamily, XGBoostFamily

    assert isinstance(
        make_model_family("xgboost", _XGB_PARAMS, N_FEATURES), XGBoostFamily
    )
    assert isinstance(
        make_model_family("mlp", _TORCH_PARAMS["mlp"], N_FEATURES), TorchFamily
    )
    with pytest.raises(ValueError, match="Unknown model family"):
        make_model_family("random_forest", {}, N_FEATURES)


def test_torch_family_efficiency_requires_timed_predict():
    X_tr, y_tr = _tabular_arrays()
    X_te, y_te = _tabular_arrays(24)
    train, val, test = (
        _loader(X_tr, y_tr),
        _loader(X_tr, y_tr),
        _loader(X_te, y_te),
    )

    fam = make_model_family(
        "mlp", _TORCH_PARAMS["mlp"], N_FEATURES, device="cpu", max_epochs=1
    )
    fam.fit(train, val)

    # Untimed predict leaves no inference measurement.
    fam.predict(test)
    with pytest.raises(RuntimeError, match="timed predict"):
        fam.efficiency(n_train_samples=64, n_infer_samples=24)

    # A timed predict enables the efficiency record.
    fam.predict(test, timed=True)
    eff = fam.efficiency(n_train_samples=64, n_infer_samples=24)
    assert eff["n_params"] > 0
    assert eff["infer_time_s"] >= 0.0


def test_torch_family_matches_inline_recipe():
    """The adapter must reproduce run_axis1's inline train/eval bit-for-bit."""
    from test_model_factory import _MockTrial  # reuse the params shim

    from lift_nids.training.early_stopping import EarlyStopping
    from lift_nids.training.inference import predict_pytorch
    from lift_nids.training.optuna_search import _suggest_pytorch_model
    from lift_nids.training.trainer import Trainer, build_optimizer, compute_class_weights
    from lift_nids.utils.reproducibility import set_global_seed

    params = _TORCH_PARAMS["mlp"]
    X_tr, y_tr = _tabular_arrays(64)
    X_vl, y_vl = _tabular_arrays(32)
    X_te, y_te = _tabular_arrays(24)
    train = _loader(X_tr, y_tr, 16, shuffle=True)
    val = _loader(X_vl, y_vl, 16)
    test = _loader(X_te, y_te, 16)

    # -- Reference: the inline recipe the runner used --------------------
    set_global_seed(7)
    ref_model, lr, wd, _ = _suggest_pytorch_model(_MockTrial(params), "mlp", N_FEATURES, 2)
    ref_model = ref_model.to("cpu")
    labels = np.concatenate([y.cpu().numpy() for _, y in train])
    weights = compute_class_weights(labels, 2).to("cpu")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(ref_model, "adam", lr=lr, weight_decay=wd)
    es = EarlyStopping(patience=5, restore_best=True)
    Trainer(ref_model, optimizer, criterion, device="cpu", early_stopping=es).fit(
        train, val, max_epochs=3
    )
    ref_true, ref_pred, ref_proba = predict_pytorch(ref_model, test, "cpu")

    # -- Adapter: same seed, same result --------------------------------
    set_global_seed(7)
    fam = make_model_family(
        "mlp", params, N_FEATURES, device="cpu", patience=5, max_epochs=3,
        use_class_weights=True, log_train_metrics=True,
    )
    fam.fit(train, val)
    pred = fam.predict(test)

    np.testing.assert_array_equal(pred.y_true, ref_true)
    np.testing.assert_array_equal(pred.y_pred, ref_pred)
    np.testing.assert_allclose(pred.y_proba, ref_proba, rtol=0, atol=0)
