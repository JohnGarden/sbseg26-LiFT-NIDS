"""XGBoost wrapper compatible with the LiFT-NIDS training pipeline."""

from __future__ import annotations

import inspect
import time
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

_DEVICE_MISMATCH_PATTERN = "Falling back to prediction using DMatrix"


class XGBoostWrapper:
    """Thin wrapper around XGBClassifier that matches the Trainer interface.

    Accepts tabular flow vectors (N, n_features) and produces binary predictions.
    Tracks train_time_s, infer_time_s, and n_params after fitting.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "logloss",
        seed: int = 42,
        scale_pos_weight: float = 1.0,
        n_jobs: int = -1,
        device: str = "cpu",
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        # XGBoost >= 2.0 uses device="cuda"; older API used tree_method="gpu_hist"
        xgb_device = "cuda" if "cuda" in device else "cpu"
        self._device = xgb_device
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric=eval_metric,
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs,
            device=xgb_device,
        )

    @classmethod
    def from_dict(cls, params: dict) -> XGBoostWrapper:
        """Construct from a hyperparameter dictionary.

        Raises TypeError for unknown keys (mirrors __init__ behaviour).
        """
        valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
        unknown = set(params) - valid
        if unknown:
            raise TypeError(f"Unknown hyperparameter(s): {unknown}")
        return cls(**params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> XGBoostWrapper:
        """Train model. If X_val is provided, uses it for early stopping."""
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        t0 = time.perf_counter()
        self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.train_time_s: float = time.perf_counter() - t0
        self.n_params: int = self._n_estimators * self._max_depth
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self._device == "cuda":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_DEVICE_MISMATCH_PATTERN)
                result = self._model.predict(X)
        else:
            result = self._model.predict(X)
        self.infer_time_s: float = time.perf_counter() - t0
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of the positive class, shape (N,)."""
        t0 = time.perf_counter()
        if self._device == "cuda":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_DEVICE_MISMATCH_PATTERN)
                result = self._model.predict_proba(X)[:, 1]
        else:
            result = self._model.predict_proba(X)[:, 1]
        self.infer_time_s = time.perf_counter() - t0
        return result

    def save(self, path: str | Path) -> None:
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> XGBoostWrapper:
        obj = cls.__new__(cls)
        obj._model = XGBClassifier()
        obj._model.load_model(str(path))
        obj._device = "cpu"
        return obj
