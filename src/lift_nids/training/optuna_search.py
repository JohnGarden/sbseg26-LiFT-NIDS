"""Optuna hyperparameter search for all model families."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm as _tqdm

from lift_nids.evaluation.metrics import (  # noqa: PLC2701
    _extract_f1_plus,
    compute_classification_metrics,
)
from lift_nids.models.factory import CNN_CHANNEL_CONFIGS, MLP_HIDDEN_CONFIGS, build_torch_model
from lift_nids.models.xgboost_wrapper import XGBoostWrapper
from lift_nids.training.early_stopping import EarlyStopping
from lift_nids.training.trainer import Trainer, build_optimizer, compute_class_weights

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_tqdm_callback(pbar: _tqdm) -> Any:
    """Return an Optuna callback that updates a tqdm bar after each trial."""
    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        dur = trial.duration.total_seconds() if trial.duration else 0
        postfix: dict[str, str] = {"trial": str(trial.number)}
        try:
            postfix["best_F1"] = f"{study.best_value:.4f}"
        except ValueError:
            postfix["best_F1"] = "—"
        postfix["last"] = f"{dur:.0f}s"
        pbar.set_postfix(postfix, refresh=True)
        pbar.update(1)
    return callback


def run_optuna_search(
    model_type: str,
    train_data: DataLoader | Dataset | tuple[np.ndarray, np.ndarray],
    val_data: DataLoader | Dataset | tuple[np.ndarray, np.ndarray],
    n_features: int,
    n_classes: int = 2,
    n_trials: int = 50,
    max_epochs: int = 30,
    patience: int = 10,
    study_name: str | None = None,
    storage: str | None = None,
    seed: int = 42,
    timeout: int | None = None,
    device: str = "cpu",
    use_class_weights: bool = False,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search for the given model type.

    Args:
        model_type: One of "xgboost", "mlp", "cnn_bilstm", "transformer".
        train_data: DataLoader or Dataset (PyTorch models) or (X, y) tuple (XGBoost).
            Pass a Dataset instead of a DataLoader to enable batch_size search.
        val_data: Same format as train_data.
        n_features: Number of input features.
        n_classes: Number of output classes (default 2).
        n_trials: Number of Optuna trials (default 50).
        max_epochs: Maximum training epochs per trial (PyTorch models).
        patience: Early stopping patience per trial (default 10).
        study_name: Optuna study name (auto-generated if None).
        storage: SQLite URI, e.g. "sqlite:///results/optuna.db". The full study
            is persisted so runs are reproducible and resumable.
        seed: Random seed for Optuna TPE sampler.
        timeout: Wall-clock limit in seconds (None = unlimited).
        device: Torch device string.
        use_class_weights: Apply inverse-frequency class weights in criterion.

    Returns:
        Dict with keys: best_params, best_value, study_name, n_trials_completed.
    """
    study_name = study_name or f"lift_nids_{model_type}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    finished = [t for t in study.trials if t.state.is_finished()]
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining_trials = max(0, n_trials - len(finished))

    if remaining_trials == 0 and complete:
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study_name": study_name,
            "n_trials_completed": len(study.trials),
        }

    objective = _make_objective(
        model_type=model_type,
        train_data=train_data,
        val_data=val_data,
        n_features=n_features,
        n_classes=n_classes,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        use_class_weights=use_class_weights,
    )

    if show_progress_bar:
        pbar = _tqdm(
            total=remaining_trials,
            initial=0,
            desc=f"Optuna [{model_type}]",
            unit="trial",
            dynamic_ncols=True,
        )
        callbacks: list[Any] = [_make_tqdm_callback(pbar)]
    else:
        pbar = None
        callbacks = []

    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=timeout,
            show_progress_bar=False,
            callbacks=callbacks or None,
        )
    finally:
        if pbar is not None:
            pbar.close()

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study_name": study_name,
        "n_trials_completed": len(study.trials),
    }


# ------------------------------------------------------------------
# Objective factory
# ------------------------------------------------------------------

def _make_objective(
    model_type: str,
    train_data: Any,
    val_data: Any,
    n_features: int,
    n_classes: int,
    max_epochs: int,
    patience: int,
    device: str,
    use_class_weights: bool,
):
    """Return an Optuna objective closure for the given model type."""

    def objective(trial: optuna.Trial) -> float:
        if model_type == "xgboost":
            return _xgboost_objective(trial, train_data, val_data, device=device)

        model, lr, weight_decay, batch_size = _suggest_pytorch_model(
            trial, model_type, n_features, n_classes
        )

        # Build DataLoaders — if caller passed Dataset objects, we create loaders
        # here so that batch_size is part of the search space.
        train_loader, val_loader = _resolve_loaders(train_data, val_data, batch_size)

        # Class weights for imbalanced data (e.g. CICIoT2023)
        criterion: nn.Module
        if use_class_weights:
            all_labels = np.concatenate([y.cpu().numpy() for _, y in train_loader])
            weights = compute_class_weights(all_labels, n_classes).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = build_optimizer(model, optimizer_name="adam", lr=lr, weight_decay=weight_decay)
        es = EarlyStopping(patience=patience, restore_best=True)
        trainer = Trainer(
            model, optimizer, criterion,
            device=device, early_stopping=es,
            n_classes=n_classes, monitor="f1_1",
        )

        def pruning_callback(epoch: int, val_metrics: dict) -> bool:
            trial.report(val_metrics.get("f1_1", 0.0), epoch)
            return trial.should_prune()

        history = trainer.fit(
            train_loader, val_loader,
            max_epochs=max_epochs,
            epoch_callback=pruning_callback,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

        return max(history["val_f1_1"]) if history["val_f1_1"] else 0.0

    return objective


def _resolve_loaders(
    train_data: Any,
    val_data: Any,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) with the trial's batch_size applied.

    Wraps Dataset inputs in DataLoaders with the suggested batch_size.
    Pre-built DataLoaders are rebuilt with the trial's batch_size so that
    the suggested batch_size is actually used during HPO (not just stored in
    best_params but never tested).
    """
    def _wrap(data: Any, shuffle: bool) -> DataLoader:
        if isinstance(data, Dataset):
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        if isinstance(data, DataLoader):
            # Rebuild with the trial's batch_size so it is applied during HPO.
            # Preserves num_workers and pin_memory from the original loader.
            nw = data.num_workers
            return DataLoader(
                data.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=nw,
                pin_memory=data.pin_memory,
                persistent_workers=nw > 0,
            )
        return data

    return _wrap(train_data, shuffle=True), _wrap(val_data, shuffle=False)


def _xgboost_objective(
    trial: optuna.Trial,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    device: str = "cpu",
) -> float:
    X_train, y_train = train_data
    X_val, y_val = val_data

    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
    }
    model = XGBoostWrapper(**params, device=device)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Objective: F1+ (positive-class F1), matching the PyTorch objective and the
    # project methodology.  Never falls back to f1_macro so HPO is consistent
    # across all model families.
    preds = model.predict(X_val)
    return _extract_f1_plus(compute_classification_metrics(y_val, preds))


def _suggest_pytorch_model(
    trial: optuna.Trial,
    model_type: str,
    n_features: int,
    n_classes: int,
) -> tuple[nn.Module, float, float, int]:
    """Suggest hyperparameters and return (model, lr, weight_decay, batch_size).

    Suggests the training and architecture hyperparameters via ``trial`` and then
    delegates object construction to :func:`build_torch_model`, so HPO and refit
    build models through exactly the same path. The order of ``trial.suggest_*``
    calls is load-bearing — it defines the Optuna sampler's parameter sequence —
    and must not change.
    """
    params: dict[str, Any] = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
    }

    if model_type == "mlp":
        params["hidden_config_idx"] = trial.suggest_int(
            "hidden_config_idx", 0, len(MLP_HIDDEN_CONFIGS) - 1
        )
        params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        params["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])
        params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    elif model_type == "cnn_bilstm":
        params["channel_config_idx"] = trial.suggest_int(
            "channel_config_idx", 0, len(CNN_CHANNEL_CONFIGS) - 1
        )
        params["lstm_hidden_size"] = trial.suggest_int("lstm_hidden_size", 64, 256)
        params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    elif model_type == "transformer":
        # Architecture FIXED (IM09) — only dropout and training HPs vary
        params["dropout"] = trial.suggest_float("dropout", 0.05, 0.3)
        params["head_dropout"] = trial.suggest_float("head_dropout", 0.05, 0.3)
        params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    else:
        raise ValueError(f"Unknown model_type for PyTorch: {model_type!r}")

    return build_torch_model(model_type, params, n_features, n_classes)
