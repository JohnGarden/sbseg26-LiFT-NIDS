"""One fit / predict / efficiency seam over all model families.

Callers used to branch on the model family at every stage — construction
(``_suggest_pytorch_model`` vs ``XGBoostWrapper``), training (``Trainer`` vs
``.fit``), prediction (``predict_pytorch`` vs ``.predict``/``.predict_proba``),
and efficiency (``count_trainable_params`` vs wrapper attributes). This module
puts that behaviour behind a single :class:`ModelFamily` interface with two
adapters:

* :class:`XGBoostFamily` — the sklearn-style tabular ensemble, and
* :class:`TorchFamily` — the ``Trainer``-driven neural models (MLP,
  CNN-BiLSTM, LightTransformer).

The adapters reproduce the exact recipe the runners used, so results are
numerically unchanged; the leverage is that a caller selects a family once via
:func:`make_model_family` and then speaks the same three verbs to any of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.nn as nn

from lift_nids.evaluation.efficiency import (
    build_efficiency_record,
    count_trainable_params,
    measure_pytorch_inference,
)
from lift_nids.models.factory import build_torch_model
from lift_nids.models.xgboost_wrapper import XGBoostWrapper
from lift_nids.training.early_stopping import EarlyStopping
from lift_nids.training.inference import predict_pytorch
from lift_nids.training.trainer import Trainer, build_optimizer, compute_class_weights

_TORCH_FAMILIES = frozenset({"mlp", "cnn_bilstm", "transformer"})


@dataclass
class Prediction:
    """Aligned evaluation arrays for one test split."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


class ModelFamily(ABC):
    """Train and evaluate one model family behind a uniform interface.

    Inputs are the family's native form — DataLoaders for :class:`TorchFamily`,
    ``(X, y)`` array tuples for :class:`XGBoostFamily` — because that split is
    inherent to the family and the caller already prepared the matching inputs.
    Everything downstream of that (the training recipe, prediction, and the
    efficiency record) is uniform.
    """

    @abstractmethod
    def fit(self, train: Any, val: Any) -> None:
        """Train the model on ``train``, using ``val`` for early stopping."""

    @abstractmethod
    def predict(self, test: Any, *, timed: bool = False) -> Prediction:
        """Evaluate on ``test``. When ``timed``, record inference wall-clock."""

    @abstractmethod
    def efficiency(self, *, n_train_samples: int, n_infer_samples: int) -> dict[str, Any]:
        """Build the standardized efficiency record for the last fit/predict."""


class XGBoostFamily(ModelFamily):
    """Adapter over :class:`XGBoostWrapper` (tabular, numpy arrays)."""

    def __init__(self, params: dict, *, seed: int, device: str) -> None:
        self._device = device
        self._model = XGBoostWrapper(**params, seed=seed, device=device)
        self._n_train = 0

    def fit(self, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray]) -> None:
        X_train, y_train = train
        X_val, y_val = val
        self._model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        self._n_train = len(X_train)

    def predict(
        self, test: tuple[np.ndarray, np.ndarray], *, timed: bool = False
    ) -> Prediction:
        X_test, y_test = test
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)  # sets infer_time_s
        return Prediction(y_true=np.asarray(y_test), y_pred=y_pred, y_proba=y_proba)

    def efficiency(self, *, n_train_samples: int, n_infer_samples: int) -> dict[str, Any]:
        return build_efficiency_record(
            n_params=self._model.n_params,
            train_time_s=self._model.train_time_s,
            infer_time_s=self._model.infer_time_s,
            n_train_samples=n_train_samples,
            n_infer_samples=n_infer_samples,
            peak_gpu_mem_bytes=0,
            device=self._device,
        )


class TorchFamily(ModelFamily):
    """Adapter over the ``Trainer``-driven neural models (DataLoaders)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        weight_decay: float,
        device: str,
        n_classes: int = 2,
        patience: int = 5,
        max_epochs: int = 30,
        use_class_weights: bool = True,
        log_train_metrics: bool = True,
        desc: str | None = None,
    ) -> None:
        self._model = model.to(device)
        self._lr = lr
        self._weight_decay = weight_decay
        self._device = device
        self._n_classes = n_classes
        self._patience = patience
        self._max_epochs = max_epochs
        self._use_class_weights = use_class_weights
        self._log_train_metrics = log_train_metrics
        self._desc = desc
        self._history: dict[str, Any] | None = None
        self._infer_time_s: float | None = None

    def fit(self, train: Any, val: Any) -> None:
        if self._use_class_weights:
            labels = np.concatenate([y.cpu().numpy() for _, y in train])
            weights = compute_class_weights(labels, self._n_classes).to(self._device)
            criterion: nn.Module = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = build_optimizer(
            self._model, "adam", lr=self._lr, weight_decay=self._weight_decay
        )
        early_stopping = EarlyStopping(patience=self._patience, restore_best=True)
        trainer = Trainer(
            self._model,
            optimizer,
            criterion,
            device=self._device,
            early_stopping=early_stopping,
            n_classes=self._n_classes,
            log_train_metrics=self._log_train_metrics,
        )
        self._history = trainer.fit(
            train, val, max_epochs=self._max_epochs, desc=self._desc
        )

    def predict(self, test: Any, *, timed: bool = False) -> Prediction:
        if timed:
            y_true, y_pred, y_proba, self._infer_time_s = measure_pytorch_inference(
                predict_pytorch, self._model, test, self._device
            )
        else:
            y_true, y_pred, y_proba = predict_pytorch(self._model, test, self._device)
        return Prediction(y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    def efficiency(self, *, n_train_samples: int, n_infer_samples: int) -> dict[str, Any]:
        if self._history is None:
            raise RuntimeError("efficiency() requires fit() to have run first")
        if self._infer_time_s is None:
            raise RuntimeError("efficiency() requires a timed predict() first")
        return build_efficiency_record(
            n_params=count_trainable_params(self._model),
            train_time_s=self._history["train_time_s"],
            infer_time_s=self._infer_time_s,
            n_train_samples=n_train_samples,
            n_infer_samples=n_infer_samples,
            peak_gpu_mem_bytes=self._history["peak_gpu_mem_bytes"],
            device=self._device,
        )


def make_model_family(
    family: str,
    params: dict,
    n_features: int,
    *,
    n_classes: int = 2,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 5,
    max_epochs: int = 30,
    use_class_weights: bool = True,
    log_train_metrics: bool = True,
    desc: str | None = None,
) -> ModelFamily:
    """Build the adapter for ``family`` from stored hyperparameters.

    For neural families the model is constructed here (consuming the RNG for
    weight init), so callers should seed immediately before calling this to keep
    per-seed runs reproducible — matching the order the runners used.

    Args:
        family: One of ``"xgboost"``, ``"mlp"``, ``"cnn_bilstm"``, ``"transformer"``.
        params: Stored ``best_params`` for the family.
        n_features: Number of input features.
        n_classes: Number of output classes.
        device: Torch device string; also the XGBoost device.
        seed: Seed passed to :class:`XGBoostWrapper` (ignored by neural models,
            which read the global RNG state).
        patience, max_epochs, use_class_weights, log_train_metrics, desc:
            Neural training-recipe knobs (ignored for XGBoost).

    Returns:
        A configured :class:`ModelFamily`.

    Raises:
        ValueError: If ``family`` is unknown.
    """
    if family == "xgboost":
        return XGBoostFamily(params, seed=seed, device=device)

    if family in _TORCH_FAMILIES:
        model, lr, weight_decay, _batch_size = build_torch_model(
            family, params, n_features, n_classes
        )
        return TorchFamily(
            model,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            n_classes=n_classes,
            patience=patience,
            max_epochs=max_epochs,
            use_class_weights=use_class_weights,
            log_train_metrics=log_train_metrics,
            desc=desc,
        )

    raise ValueError(f"Unknown model family: {family!r}")
