#!/usr/bin/env python3
"""CLI to run a single LiFT-NIDS experiment from a YAML config.

Usage:
    uv run python scripts/run_experiment.py \
        --config configs/experiments/exp_001_ciciot_static_xgb.yaml
    uv run python scripts/run_experiment.py --config ... --sample-frac 0.05 --n-trials 10
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml

# Ensure src/ is on path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lift_nids.data.dataloader import build_dataloader
from lift_nids.data.dataset import FlowDataset, LazySequenceFlowDataset
from lift_nids.data.preprocessing import FlowPreprocessor, infer_feature_types
from lift_nids.data.splits import (
    forward_chaining_splits,
    static_split,
    temporal_train_val_test_split,
)
from lift_nids.evaluation.efficiency import (
    aggregate_efficiency_records,
    build_efficiency_record,
    count_trainable_params,
    measure_pytorch_inference,
)
from lift_nids.evaluation.metrics import (  # noqa: PLC2701
    _extract_f1_plus,
    compute_classification_metrics,
)
from lift_nids.evaluation.temporal_metrics import compute_naut, median_trajectory
from lift_nids.experiments.results import provenance, write_result
from lift_nids.models.factory import build_torch_model
from lift_nids.training.inference import predict_pytorch
from lift_nids.training.optuna_search import run_optuna_search
from lift_nids.utils.logging import log_experiment_result, setup_logger
from lift_nids.utils.reproducibility import set_global_seed

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(exp_config_path: str | Path) -> dict:
    """Load experiment config merged with all base configs."""
    exp = _load_yaml(exp_config_path)
    merged: dict = {}
    for base_path in exp.get("experiment", {}).get("base_configs", []):
        merged = _deep_merge(merged, _load_yaml(base_path))
    dataset_cfg = exp.get("experiment", {}).get("dataset_config")
    if dataset_cfg:
        merged = _deep_merge(merged, _load_yaml(dataset_cfg))
    model_cfg = exp.get("experiment", {}).get("model_config")
    if model_cfg:
        merged = _deep_merge(merged, _load_yaml(model_cfg))
    merged = _deep_merge(merged, exp)
    return merged


# ---------------------------------------------------------------------------
# Device and GPU diagnostics
# ---------------------------------------------------------------------------

def resolve_device(config: dict, device_override: str | None = None) -> str:
    """Resolve the requested training device and fail fast for invalid CUDA use."""
    requested = device_override or config.get("training", {}).get("device", "auto")
    requested = str(requested).lower()

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Run `python scripts/run_experiment.py --gpu-check` for diagnostics."
        )

    return requested


def collect_torch_cuda_info() -> dict[str, Any]:
    """Return CUDA metadata reported by the local PyTorch installation."""
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "devices": [],
    }

    if not torch.cuda.is_available():
        return info

    info["current_device"] = torch.cuda.current_device()
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info["devices"].append(
            {
                "index": idx,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return info


def _normalize_cuda_device(device: str) -> torch.device:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(f"GPU diagnostics require a CUDA device, got {device!r}.")
    if torch_device.index is not None:
        torch.cuda.set_device(torch_device)
        return torch_device
    return torch.device("cuda", torch.cuda.current_device())


def test_torch_cuda_tensor_ops(device: str = "cuda") -> dict[str, Any]:
    """Smoke-test CUDA allocation, matrix math, backward pass, and sync."""
    result: dict[str, Any] = {"name": "torch_cuda_tensor_ops", "ok": False, "device": device}
    if not torch.cuda.is_available():
        result["error"] = "torch.cuda.is_available() is False"
        return result

    try:
        torch_device = _normalize_cuda_device(device)
        torch.cuda.empty_cache()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x = torch.randn((512, 512), device=torch_device, requires_grad=True)
        w = torch.randn((512, 512), device=torch_device)
        loss = (x @ w).relu().mean()
        loss.backward()

        end.record()
        torch.cuda.synchronize(torch_device)
        result.update(
            {
                "ok": True,
                "elapsed_ms": round(start.elapsed_time(end), 3),
                "loss": float(loss.detach().cpu()),
                "x_device": str(x.device),
                "grad_device": str(x.grad.device) if x.grad is not None else None,
                "allocated_bytes": torch.cuda.memory_allocated(torch_device),
                "reserved_bytes": torch.cuda.memory_reserved(torch_device),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def test_cuda_dataloader_transfer(device: str = "cuda") -> dict[str, Any]:
    """Smoke-test pinned DataLoader batches and non-blocking transfer to CUDA."""
    result: dict[str, Any] = {
        "name": "cuda_dataloader_transfer",
        "ok": False,
        "device": device,
    }
    if not torch.cuda.is_available():
        result["error"] = "torch.cuda.is_available() is False"
        return result

    try:
        from torch.utils.data import DataLoader, TensorDataset

        torch_device = _normalize_cuda_device(device)
        X = torch.randn(4096, 32)
        y = torch.randint(0, 2, (4096,))
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=1024,
            pin_memory=True,
            shuffle=False,
        )
        X_batch, y_batch = next(iter(loader))
        X_gpu = X_batch.to(torch_device, non_blocking=True)
        y_gpu = y_batch.to(torch_device, non_blocking=True)
        torch.cuda.synchronize(torch_device)
        result.update(
            {
                "ok": True,
                "batch_shape": list(X_gpu.shape),
                "input_was_pinned": X_batch.is_pinned(),
                "x_device": str(X_gpu.device),
                "y_device": str(y_gpu.device),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def test_xgboost_cuda_smoke() -> dict[str, Any]:
    """Smoke-test whether XGBoost accepts and runs with device='cuda'."""
    result: dict[str, Any] = {"name": "xgboost_cuda_smoke", "ok": False, "device": "cuda"}
    if not torch.cuda.is_available():
        result["error"] = "torch.cuda.is_available() is False"
        return result

    try:
        from xgboost import XGBClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(size=(256, 12)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = XGBClassifier(
                n_estimators=8,
                max_depth=2,
                learning_rate=0.2,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                device="cuda",
                verbosity=1,
            )
            model.fit(X, y, verbose=False)
            proba_1d = model.get_booster().predict(xgb.DMatrix(X[:8]))
            proba = np.column_stack([1 - proba_1d, proba_1d])

        warning_messages = [str(item.message) for item in caught]
        fallback_markers = (
            "No visible GPU",
            "not compiled with GPU",
            "falling back to CPU",
            "must have at least one device",
        )
        fallback_warning = next(
            (
                msg
                for msg in warning_messages
                if any(marker.lower() in msg.lower() for marker in fallback_markers)
            ),
            None,
        )

        result.update(
            {
                "ok": fallback_warning is None,
                "proba_shape": list(proba.shape),
                "warnings": warning_messages,
            }
        )
        if fallback_warning is not None:
            result["error"] = fallback_warning
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def run_gpu_diagnostics(
    device: str = "cuda",
    include_xgboost: bool = True,
) -> dict[str, Any]:
    """Run local GPU checks without loading datasets or starting an experiment."""
    checks = [
        test_torch_cuda_tensor_ops(device=device),
        test_cuda_dataloader_transfer(device=device),
    ]
    if include_xgboost:
        checks.append(test_xgboost_cuda_smoke())

    return {
        "torch_cuda": collect_torch_cuda_info(),
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Columns to always exclude from feature sets (labels + metadata from both datasets).
# CICIoT2023 uses lowercase; MAWIFlow (CICFlowMeter) uses title-case with spaces.
# Both 'timestamp' and 'Timestamp' are excluded so that the legacy CICFlowMeter
# column name and the canonical normalized name are both covered.
_LABEL_COLS = {
    # CICIoT2023
    "label", "label_multiclass",
    "flow_id", "timestamp",
    # MAWIFlow
    "Label", "taxonomy", "label_1",
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    # Added by load_dataset for temporal splits
    "year",
    # Added by build/process MAWIFlow scripts as join key (excluded from output, but
    # present in any intermediate DataFrames loaded before the EXCLUDE step)
    "_flow_row_id",
}


def _resolve_timestamp_col(df: pd.DataFrame, cfg: dict) -> str | None:
    """Resolve the canonical timestamp column name from config with legacy fallback.

    Priority:
    1. Column named by ``dataset.timestamp_column`` in config (canonical name).
    2. ``'Timestamp'`` — CICFlowMeter title-case, accepted only as legacy input.

    After ``load_dataset`` normalizes the DataFrame, the canonical name is
    always used downstream; callers should invoke this function once (on the
    loaded DataFrame) and then pass the result to all temporal split calls.

    Args:
        df: DataFrame to inspect.
        cfg: Experiment config dict with optional ``dataset.timestamp_column``.

    Returns:
        Column name as it appears in *df*, or ``None`` if absent.
    """
    canonical = cfg.get("dataset", {}).get("timestamp_column", "timestamp")
    if canonical in df.columns:
        return canonical
    if "Timestamp" in df.columns:
        return "Timestamp"
    return None


def _validate_and_normalize_timestamp(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, str | None]:
    """Normalize the timestamp column and raise on invalid values.

    Resolves the canonical column name, renames the CICFlowMeter legacy column
    ``'Timestamp'`` to the canonical name, converts to ``datetime64``, and
    raises ``ValueError`` if any values fail to parse (NaT after coercion).

    For temporal datasets — detected by the presence of a ``'year'`` column,
    which ``load_dataset`` adds for MAWIFlow — a missing timestamp column is
    also a hard error.  For non-temporal datasets (CICIoT2023 static splits),
    an absent timestamp column is silently allowed.

    Args:
        df: DataFrame after all files have been loaded and concatenated.
        cfg: Experiment config dict with optional ``dataset.timestamp_column``.

    Returns:
        ``(df, canonical_col_name)`` with the validated and renamed column, or
        ``(df, None)`` if no timestamp column is present (non-temporal only).

    Raises:
        ValueError: If a temporal dataset is missing its timestamp column, or
            if any rows contain unparseable timestamp values.
    """
    ds_cfg = cfg.get("dataset", {})
    canonical = ds_cfg.get("timestamp_column", "timestamp")
    is_temporal = "year" in df.columns

    found = _resolve_timestamp_col(df, cfg)
    if found is None:
        if is_temporal:
            raise ValueError(
                f"Timestamp column '{canonical}' is required for temporal datasets "
                f"(MAWIFlow 'year' column detected) but was not found. "
                f"Ensure each yearly parquet file contains a timestamp column. "
                f"Config key: dataset.timestamp_column={canonical!r}."
            )
        return df, None

    if found != canonical:
        df = df.rename(columns={found: canonical})

    original_values = df[canonical].copy()
    df[canonical] = pd.to_datetime(df[canonical], errors="coerce")
    nat_mask = df[canonical].isna()
    nat_count = int(nat_mask.sum())

    if nat_count > 0:
        invalid_examples = original_values[nat_mask].head(5).tolist()
        raise ValueError(
            f"Timestamp column '{canonical}' contains {nat_count} value(s) that "
            f"could not be parsed as datetime (NaT after coercion). "
            f"Sample invalid values: {invalid_examples}. "
            f"Fix the parquet/preprocessing pipeline to ensure all timestamps are "
            f"valid datetime strings before running experiments."
        )

    return df, canonical


def _compute_num_workers() -> int:
    """Return a safe number of DataLoader workers for the current machine.

    Caps at 4 to avoid excessive memory usage with large in-memory datasets.
    Windows uses spawn-based multiprocessing, which works but is slower to start
    than fork — persistent_workers=True compensates for the per-epoch spawn cost.
    """
    cpu = os.cpu_count() or 1
    return min(4, max(0, cpu // 2))


def load_dataset(cfg: dict, sample_frac: float | None = None, seed: int = 42) -> pd.DataFrame:
    """Load CICIoT2023 or MAWIFlow parquet(s) from config path.

    When multiple parquet files exist (e.g. one per year for MAWIFlow), all are
    concatenated. Files whose stem is an integer get a 'year' column set to that
    integer, enabling forward-chaining splits downstream.

    The timestamp column is resolved from ``dataset.timestamp_column`` (canonical)
    with a fallback to ``'Timestamp'`` (CICFlowMeter title-case legacy). If the
    legacy name is found, the column is **renamed** to the canonical name so that
    all downstream code works with a single, consistent name.  The column is
    converted to ``datetime64`` and used for chronological sorting.  Sorting is
    applied after ``sample_frac`` so that subsampled DataFrames remain in temporal
    order.

    CICIoT2023 (single file, no ``'year'`` column, no timestamp) is unaffected by
    the sorting step.
    """
    ds_cfg = cfg.get("dataset", {})
    data_path = Path(ds_cfg.get("path", "data/raw/CICIoT2023"))
    fmt = ds_cfg.get("file_format", "parquet")

    if fmt != "parquet":
        raise ValueError(f"Unsupported file_format: {fmt!r}")

    candidates = sorted(data_path.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    if len(candidates) == 1:
        df = pd.read_parquet(candidates[0], engine="pyarrow")
    else:
        dfs = []
        for f in candidates:
            year_df = pd.read_parquet(f, engine="pyarrow")
            try:
                year_df["year"] = int(f.stem)
            except ValueError:
                pass
            dfs.append(year_df)
        df = pd.concat(dfs, ignore_index=True)

    # Validate, rename (Timestamp→timestamp), and convert the timestamp column.
    # Raises ValueError if the column is required but missing (temporal/MAWIFlow)
    # or if any values fail datetime parsing.
    df, _validated_ts = _validate_and_normalize_timestamp(df, cfg)
    canonical_ts = _validated_ts or ds_cfg.get("timestamp_column", "timestamp")

    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    # Ensure chronological order for temporal protocols.  Re-applied after
    # sample_frac so that subsampled DataFrames are still in temporal order.
    # CICIoT2023 (single file, no 'year' column) is unaffected.
    _sort_keys = [k for k in ("year", canonical_ts) if k in df.columns]
    if _sort_keys:
        df = df.sort_values(_sort_keys, kind="stable").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame, label_col: str = "label") -> tuple[list[str], list[str]]:
    """Return (numerical_cols, categorical_cols), excluding label columns."""
    exclude = _LABEL_COLS | {label_col}
    num_cols, cat_cols = infer_feature_types(df, label_col=label_col)
    num_cols = [c for c in num_cols if c not in exclude]
    cat_cols = [c for c in cat_cols if c not in exclude]
    return num_cols, cat_cols


# ---------------------------------------------------------------------------
# Build model inputs
# ---------------------------------------------------------------------------

def build_tabular_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: tuple[list[str], list[str]],
    label_col: str,
    batch_size: int = 2048,
    device: str = "cpu",
    num_workers: int = 0,
) -> tuple:
    """Preprocess and build DataLoaders for tabular models."""
    num_cols, cat_cols = feature_cols
    preprocessor = FlowPreprocessor(log_transform=True)
    X_train = preprocessor.fit_transform(train_df, num_cols, cat_cols).to_numpy(np.float32)
    X_val   = preprocessor.transform(val_df).to_numpy(np.float32)
    X_test  = preprocessor.transform(test_df).to_numpy(np.float32)

    y_train = train_df[label_col].to_numpy(np.int64)
    y_val   = val_df[label_col].to_numpy(np.int64)
    y_test  = test_df[label_col].to_numpy(np.int64)

    pin = "cuda" in device
    persistent = num_workers > 0
    train_loader = build_dataloader(
        FlowDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
        pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
    )
    val_loader  = build_dataloader(
        FlowDataset(X_val,  y_val),  batch_size=batch_size,
        pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
    )
    test_loader = build_dataloader(
        FlowDataset(X_test, y_test), batch_size=batch_size,
        pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
    )

    return (
        (X_train, y_train), (X_val, y_val), (X_test, y_test),
        train_loader, val_loader, test_loader, preprocessor,
    )


def build_tabular_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: tuple[list[str], list[str]],
    label_col: str,
) -> tuple:
    """Preprocess for tabular models and return only numpy arrays — no DataLoaders.

    Lighter-weight alternative to :func:`build_tabular_inputs` for models that
    operate directly on numpy arrays (e.g. XGBoost), avoiding DataLoader
    creation and the associated worker-spawn overhead on Windows.

    Returns:
        ``((X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor)``
    """
    num_cols, cat_cols = feature_cols
    preprocessor = FlowPreprocessor(log_transform=True)
    X_train = preprocessor.fit_transform(train_df, num_cols, cat_cols).to_numpy(np.float32)
    X_val   = preprocessor.transform(val_df).to_numpy(np.float32)
    X_test  = preprocessor.transform(test_df).to_numpy(np.float32)

    y_train = train_df[label_col].to_numpy(np.int64)
    y_val   = val_df[label_col].to_numpy(np.int64)
    y_test  = test_df[label_col].to_numpy(np.int64)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor


def build_sequence_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: tuple[list[str], list[str]],
    label_col: str,
    window_size: int = 16,
    batch_size: int = 512,
    device: str = "cpu",
    num_workers: int = 0,
) -> tuple:
    """Preprocess + apply sliding window for sequential models."""
    num_cols, cat_cols = feature_cols
    preprocessor = FlowPreprocessor(log_transform=True)
    X_train_df = preprocessor.fit_transform(train_df, num_cols, cat_cols)
    X_val_df   = preprocessor.transform(val_df)
    X_test_df  = preprocessor.transform(test_df)

    X_train_arr = X_train_df.to_numpy(np.float32)
    y_train_arr = train_df[label_col].reset_index(drop=True).to_numpy(np.int64)
    X_val_arr   = X_val_df.to_numpy(np.float32)
    y_val_arr   = val_df[label_col].reset_index(drop=True).to_numpy(np.int64)
    X_test_arr  = X_test_df.to_numpy(np.float32)
    y_test_arr  = test_df[label_col].reset_index(drop=True).to_numpy(np.int64)

    pin = "cuda" in device
    persistent = num_workers > 0
    train_loader = build_dataloader(
        LazySequenceFlowDataset(X_train_arr, y_train_arr, window_size, stride=window_size // 2),
        batch_size=batch_size, shuffle=True,
        pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
    )
    val_loader = build_dataloader(
        LazySequenceFlowDataset(X_val_arr, y_val_arr, window_size, stride=1),
        batch_size=batch_size,
        pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
    )
    test_loader = build_dataloader(
        LazySequenceFlowDataset(X_test_arr, y_test_arr, window_size, stride=1),
        batch_size=batch_size,
        pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, X_train_arr.shape[1], preprocessor


# ---------------------------------------------------------------------------
# Temporal (forward-chaining) experiment — MAWIFlow Eixo 2
# ---------------------------------------------------------------------------

def _run_temporal_experiment(
    config: dict,
    seed: int = 42,
    sample_frac: float | None = None,
    n_trials: int | None = None,
    max_epochs: int | None = None,
    device_override: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Forward-chaining temporal evaluation for MAWIFlow.

    Protocol:
    1. Single HPO on year[0] only — temporal 60/20/20 split (fit/val/internal_test).
       best_params are fixed here and reused in every subsequent window.
       internal_test is reserved and never used by Optuna.
    2. For every forward-chaining window (k from config, default k=1):
       train with best_params, evaluate on test years. No HPO inside this loop.
    3. Compute nAUT_H (H=1,3,5) and median F1+ trajectory.
    """
    import torch.nn as nn

    from lift_nids.training.early_stopping import EarlyStopping
    from lift_nids.training.trainer import Trainer, build_optimizer, compute_class_weights

    exp_name = config.get("experiment", {}).get("name", "experiment")
    logger = setup_logger(exp_name)
    set_global_seed(seed)
    logger.info("Starting temporal experiment: %s (seed=%d)", exp_name, seed)

    device = resolve_device(config, device_override)
    logger.info("Device: %s", device)

    t0 = time.time()
    logger.info("Loading MAWIFlow dataset (sample_frac=%s)…", sample_frac)
    df = load_dataset(config, sample_frac=sample_frac, seed=seed)

    year_col      = "year"
    timestamp_col = _resolve_timestamp_col(df, config)
    if timestamp_col is None:
        raise ValueError(
            "MAWIFlow temporal protocol requires an auditable timestamp column for "
            "chronological ordering, but none was found in the loaded DataFrame. "
            f"Config expects '{config.get('dataset', {}).get('timestamp_column', 'timestamp')}'. "
            "Ensure the yearly parquet files contain a valid timestamp column."
        )
    label_col = config.get("dataset", {}).get("label_column", "label")
    if year_col not in df.columns:
        raise ValueError(
            f"Column '{year_col}' not found — load_dataset must add it from the file stem. "
            "Check that data/processed/mawiflow/ contains yearly *.parquet files."
        )
    years = sorted(df[year_col].unique())
    if len(years) < 2:
        raise ValueError(f"Temporal protocol requires ≥ 2 years; got {years}")
    logger.info("Loaded %d rows | years: %s", len(df), years)

    _FAMILY_ALIASES = {"transformer_light": "transformer", "cnn_bi_lstm": "cnn_bilstm"}
    model_family = _FAMILY_ALIASES.get(
        config.get("model", {}).get("family", "mlp"),
        config.get("model", {}).get("family", "mlp"),
    )
    training_cfg = config.get("training", {})
    _max_epochs  = max_epochs or training_cfg.get("epochs", 30)
    _patience    = training_cfg.get("early_stopping", {}).get("patience", 5)
    _n_trials    = n_trials or 50
    _batch_size  = training_cfg.get("batch_size", 2048)
    window_size  = (
        config.get("experiment", {}).get("temporal_sequence", {}).get("window_size")
        or config.get("data", {}).get("temporal_sequence", {}).get("window_size", 16)
    )
    _num_workers = _compute_num_workers()

    feature_cols = get_feature_cols(df, label_col)
    n_features   = len(feature_cols[0])
    is_sequential = model_family in ("cnn_bilstm", "transformer")
    logger.info("Model: %s | n_features=%d | window_size=%d", model_family, n_features, window_size)

    storage_dir = Path(output_dir or f"data/results/{exp_name}")
    storage_dir.mkdir(parents=True, exist_ok=True)
    optuna_storage = f"sqlite:///{storage_dir}/optuna.db"

    # ── Step 1: HPO on year[0] with temporal 60/20/20 split (no shuffle) ──
    anchor_df    = df[df[year_col] == years[0]].reset_index(drop=True)
    train_hpo_df, val_hpo_df, _ = temporal_train_val_test_split(
        anchor_df, year_col, timestamp_col=timestamp_col
    )
    logger.info(
        "HPO anchor year=%d: fit=%d  val=%d  (temporal 60/20/20; internal test reserved)",
        years[0], len(train_hpo_df), len(val_hpo_df),
    )

    if is_sequential:
        hpo_train, hpo_val, _, n_features, _ = build_sequence_inputs(
            train_hpo_df, val_hpo_df, val_hpo_df,
            feature_cols, label_col,
            window_size=window_size, batch_size=_batch_size, device=device,
            num_workers=_num_workers,
        )
        hpo_train_input, hpo_val_input = hpo_train, hpo_val
    else:
        _tab = build_tabular_inputs(
            train_hpo_df, val_hpo_df, val_hpo_df,
            feature_cols, label_col,
            batch_size=_batch_size, device=device, num_workers=_num_workers,
        )
        (X_tr_h, y_tr_h), (X_vl_h, y_vl_h) = _tab[0], _tab[1]
        train_l_h, val_l_h = _tab[3], _tab[4]
        hpo_train_input = (X_tr_h, y_tr_h) if model_family == "xgboost" else train_l_h
        hpo_val_input   = (X_vl_h, y_vl_h) if model_family == "xgboost" else val_l_h

    logger.info("Running Optuna HPO (%d trials)…", _n_trials)
    search_result = run_optuna_search(
        model_type=model_family,
        train_data=hpo_train_input,
        val_data=hpo_val_input,
        n_features=n_features,
        n_classes=2,
        n_trials=_n_trials,
        max_epochs=_max_epochs,
        patience=_patience,
        study_name=exp_name,
        storage=optuna_storage,
        seed=seed,
        device=device,
        use_class_weights=True,
    )
    best_params = search_result["best_params"]
    logger.info("Best HPO val F1: %.4f | params: %s", search_result["best_value"], best_params)

    # ── Step 2: Forward-chaining evaluation ───────────────────────────────
    k_val  = config.get("experiment", {}).get("protocol", {}).get("k", 1)
    splits = forward_chaining_splits(df, year_col=year_col, k=k_val)
    logger.info("Forward-chaining (k=%s): %d windows", k_val, len(splits))

    f1plus_records: list[tuple[int, int, float]] = []
    window_results: list[dict] = []
    window_efficiency: list[dict] = []

    for win_idx, win in enumerate(splits):
        logger.info(
            "Window %d: train=%s → test=%s",
            win_idx, win.train_years, sorted(win.test_dfs_by_year.keys()),
        )

        # 60/20/20 temporal split: fit / val / internal_test (for Δt=0 evaluation)
        fit_w_df, val_w_df, internal_test_df = temporal_train_val_test_split(
            win.train_df, year_col, timestamp_col=timestamp_col
        )

        # Use first future test year as placeholder to build the initial loaders.
        first_tst_df = win.test_dfs_by_year[sorted(win.test_dfs_by_year)[0]].reset_index(
            drop=True
        )

        def _record(test_year: int, metrics: dict, f1_plus: float) -> None:
            f1plus_records.append((win.anchor_year, test_year, f1_plus))
            window_results.append({
                "window":      win_idx,
                "train_years": win.train_years,
                "anchor_year": win.anchor_year,
                "test_year":   test_year,
                **{f"test_{k}": v for k, v in metrics.items()},
            })

        if is_sequential:
            tr_l, vl_l, _, n_features, _ = build_sequence_inputs(
                fit_w_df, val_w_df, first_tst_df,
                feature_cols, label_col,
                window_size=window_size, batch_size=_batch_size, device=device,
                num_workers=_num_workers,
            )
            model_w, lr, wd, _ = build_torch_model(
                model_family, best_params, n_features, 2
            )
            model_w = model_w.to(device)
            label_arr = np.concatenate([y.cpu().numpy() for _, y in tr_l])
            weights   = compute_class_weights(label_arr, 2).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = build_optimizer(model_w, "adam", lr=lr, weight_decay=wd)
            es        = EarlyStopping(patience=_patience, restore_best=True)
            _trainer_w = Trainer(model_w, optimizer, criterion, device=device, early_stopping=es)
            _hist_w    = _trainer_w.fit(tr_l, vl_l, max_epochs=_max_epochs)

            # Δt=0: evaluate on the held-out internal test slice (same time window
            # as training, but never seen during fitting or early-stopping).
            _, _, int_tst_l, _, _ = build_sequence_inputs(
                fit_w_df, val_w_df, internal_test_df,
                feature_cols, label_col,
                window_size=window_size, batch_size=_batch_size, device=device,
                num_workers=_num_workers,
            )
            y_true_int, y_pred_int, y_proba_int, _infer_int_s = measure_pytorch_inference(
                predict_pytorch, model_w, int_tst_l, device
            )
            window_efficiency.append(build_efficiency_record(
                n_params=count_trainable_params(model_w),
                train_time_s=_hist_w["train_time_s"],
                infer_time_s=_infer_int_s,
                n_train_samples=len(fit_w_df),
                n_infer_samples=len(y_true_int),
                peak_gpu_mem_bytes=_hist_w["peak_gpu_mem_bytes"],
                device=device,
            ))
            metrics_int = compute_classification_metrics(
                y_true_int, y_pred_int, y_proba=y_proba_int
            )
            f1_plus_int = _extract_f1_plus(metrics_int)
            _record(win.anchor_year, metrics_int, f1_plus_int)
            logger.info(
                "Window %d / dt=0 (internal): F1+=%.4f  F1_macro=%.4f",
                win_idx, f1_plus_int, metrics_int.get("f1_macro", 0.0),
            )

            for test_yr, test_yr_raw_df in sorted(win.test_dfs_by_year.items()):
                tst_yr_df = test_yr_raw_df.reset_index(drop=True)
                _, _, te_yr_l, _, _ = build_sequence_inputs(
                    fit_w_df, val_w_df, tst_yr_df,
                    feature_cols, label_col,
                    window_size=window_size, batch_size=_batch_size, device=device,
                    num_workers=_num_workers,
                )
                y_true_yr, y_pred_yr, y_proba_yr = predict_pytorch(model_w, te_yr_l, device)
                metrics_yr = compute_classification_metrics(
                    y_true_yr, y_pred_yr, y_proba=y_proba_yr
                )
                f1_plus_yr = _extract_f1_plus(metrics_yr)
                _record(test_yr, metrics_yr, f1_plus_yr)
                logger.info(
                    "Window %d / test_year %d: F1+=%.4f  F1_macro=%.4f",
                    win_idx, test_yr, f1_plus_yr, metrics_yr.get("f1_macro", 0.0),
                )

        else:
            _tab_w = build_tabular_inputs(
                fit_w_df, val_w_df, first_tst_df,
                feature_cols, label_col,
                batch_size=_batch_size, device=device, num_workers=_num_workers,
            )
            (X_tr_w, y_tr_w), (X_vl_w, y_vl_w) = _tab_w[0], _tab_w[1]
            tr_l_w, vl_l_w = _tab_w[3], _tab_w[4]
            n_feat_w = X_tr_w.shape[1]

            if model_family == "xgboost":
                from lift_nids.models.xgboost_wrapper import XGBoostWrapper
                model_w = XGBoostWrapper(**best_params, device=device)
                model_w.fit(X_tr_w, y_tr_w, X_val=X_vl_w, y_val=y_vl_w)
            else:
                model_w, lr, wd, _ = build_torch_model(
                    model_family, best_params, n_feat_w, 2
                )
                model_w = model_w.to(device)
                label_arr = np.concatenate([y.cpu().numpy() for _, y in tr_l_w])
                weights   = compute_class_weights(label_arr, 2).to(device)
                criterion = nn.CrossEntropyLoss(weight=weights)
                optimizer = build_optimizer(model_w, "adam", lr=lr, weight_decay=wd)
                es        = EarlyStopping(patience=_patience, restore_best=True)
                _trainer_tab = Trainer(model_w, optimizer, criterion, device=device, early_stopping=es)
                _hist_tab    = _trainer_tab.fit(tr_l_w, vl_l_w, max_epochs=_max_epochs)

            # Δt=0: internal test slice
            _tab_int = build_tabular_inputs(
                fit_w_df, val_w_df, internal_test_df,
                feature_cols, label_col,
                batch_size=_batch_size, device=device, num_workers=_num_workers,
            )
            (_, _), (_, _), (X_te_int, y_te_int) = _tab_int[0], _tab_int[1], _tab_int[2]
            te_int_l = _tab_int[5]

            if model_family == "xgboost":
                y_pred_int  = model_w.predict(X_te_int)
                y_proba_int = model_w.predict_proba(X_te_int)  # sets infer_time_s
                y_true_int  = y_te_int
                window_efficiency.append(build_efficiency_record(
                    n_params=model_w.n_params,
                    train_time_s=model_w.train_time_s,
                    infer_time_s=model_w.infer_time_s,
                    n_train_samples=len(X_tr_w),
                    n_infer_samples=len(X_te_int),
                    peak_gpu_mem_bytes=0,
                    device=device,
                ))
            else:
                y_true_int, y_pred_int, y_proba_int, _infer_tab_s = measure_pytorch_inference(
                    predict_pytorch, model_w, te_int_l, device
                )
                window_efficiency.append(build_efficiency_record(
                    n_params=count_trainable_params(model_w),
                    train_time_s=_hist_tab["train_time_s"],
                    infer_time_s=_infer_tab_s,
                    n_train_samples=len(X_tr_w),
                    n_infer_samples=len(y_true_int),
                    peak_gpu_mem_bytes=_hist_tab["peak_gpu_mem_bytes"],
                    device=device,
                ))

            metrics_int = compute_classification_metrics(
                y_true_int, y_pred_int, y_proba=y_proba_int
            )
            f1_plus_int = _extract_f1_plus(metrics_int)
            _record(win.anchor_year, metrics_int, f1_plus_int)
            logger.info(
                "Window %d / dt=0 (internal): F1+=%.4f  F1_macro=%.4f",
                win_idx, f1_plus_int, metrics_int.get("f1_macro", 0.0),
            )

            for test_yr, test_yr_raw_df in sorted(win.test_dfs_by_year.items()):
                tst_yr_df = test_yr_raw_df.reset_index(drop=True)
                _tab_yr = build_tabular_inputs(
                    fit_w_df, val_w_df, tst_yr_df,
                    feature_cols, label_col,
                    batch_size=_batch_size, device=device, num_workers=_num_workers,
                )
                (_, _), (_, _), (X_te_yr, y_te_yr) = _tab_yr[0], _tab_yr[1], _tab_yr[2]
                te_yr_l = _tab_yr[5]

                if model_family == "xgboost":
                    y_pred_yr  = model_w.predict(X_te_yr)
                    y_proba_yr = model_w.predict_proba(X_te_yr)
                    y_true_yr  = y_te_yr
                else:
                    y_true_yr, y_pred_yr, y_proba_yr = predict_pytorch(
                        model_w, te_yr_l, device
                    )

                metrics_yr = compute_classification_metrics(
                    y_true_yr, y_pred_yr, y_proba=y_proba_yr
                )
                f1_plus_yr = _extract_f1_plus(metrics_yr)
                _record(test_yr, metrics_yr, f1_plus_yr)
                logger.info(
                    "Window %d / test_year %d: F1+=%.4f  F1_macro=%.4f",
                    win_idx, test_yr, f1_plus_yr, metrics_yr.get("f1_macro", 0.0),
                )

    # ── Step 3: Temporal metrics ──────────────────────────────────────────
    traj   = median_trajectory(f1plus_records)
    naut_1 = compute_naut(traj, H=1)
    naut_3 = compute_naut(traj, H=3)
    naut_5 = compute_naut(traj, H=5)
    logger.info("nAUT_1=%.4f  nAUT_3=%.4f  nAUT_5=%.4f", naut_1, naut_3, naut_5)

    efficiency_summary = aggregate_efficiency_records(window_efficiency)

    elapsed = time.time() - t0
    results = {
        "experiment_name":   exp_name,
        "model_family":      model_family,
        "seed":              seed,
        "sample_frac":       sample_frac,
        "n_trials":          _n_trials,
        "k":                 k_val,
        "n_windows":         len(splits),
        "years":             years,
        "n_features":        n_features,
        "best_val_f1":              search_result["best_value"],
        "best_params":              best_params,
        # HPO policy metadata — single-anchor-year strategy (see README)
        "hpo_policy":               "single_anchor_year",
        "hpo_anchor_year":          int(years[0]),
        "hpo_split":                "temporal_60_20_20",
        "hpo_reused_across_windows": True,
        # Provenance — single-model/single-horizon auxiliary run (not the full Axis 2 matrix)
        "is_full_temporal_matrix":  False,
        "runner":                   "run_experiment.py",
        "note":                     "single-model/single-horizon auxiliary run; not the full Axis 2 matrix",
        "elapsed_seconds":          round(elapsed, 1),
        **provenance(),
        "nAUT_1":            naut_1,
        "nAUT_3":            naut_3,
        "nAUT_5":            naut_5,
        "median_trajectory": traj,
        "window_results":    window_results,
        "f1plus_records":    f1plus_records,
        "window_efficiency":    window_efficiency,
        "efficiency_summary":   efficiency_summary,
    }
    write_result(results, storage_dir / "results.json", registry_dir=Path("data/results"))
    logger.info("Done in %.1fs → %s", elapsed, storage_dir / "results.json")
    return results


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: dict,
    seed: int = 42,
    sample_frac: float | None = None,
    n_trials: int | None = None,
    max_epochs: int | None = None,
    device_override: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    # Route temporal experiments to the forward-chaining runner
    protocol = (
        config.get("experiment", {}).get("protocol", {}).get("split")
        or config.get("data", {}).get("split_protocol", "static")
    )
    if protocol == "temporal":
        return _run_temporal_experiment(
            config, seed=seed, sample_frac=sample_frac, n_trials=n_trials,
            max_epochs=max_epochs, device_override=device_override, output_dir=output_dir,
        )

    exp_name = config.get("experiment", {}).get("name", "experiment")
    logger = setup_logger(exp_name)
    set_global_seed(seed)
    logger.info("Starting experiment: %s (seed=%d)", exp_name, seed)

    # Device
    device = resolve_device(config, device_override)
    logger.info("Device: %s", device)

    # Load data
    t0 = time.time()
    logger.info("Loading dataset (sample_frac=%s)…", sample_frac)
    df = load_dataset(config, sample_frac=sample_frac, seed=seed)
    label_col = config.get("dataset", {}).get("label_column", "label")
    logger.info("Loaded %d rows. Label dist: %s", len(df), df[label_col].value_counts().to_dict())

    # Split
    split_cfg = config.get("data", {}).get("split", {})
    train_df, val_df, test_df = static_split(
        df,
        train_size=split_cfg.get("train_size", 0.60),
        val_size=split_cfg.get("val_size", 0.20),
        test_size=split_cfg.get("test_size", 0.20),
        stratify_col=label_col if split_cfg.get("stratify", True) else None,
        seed=seed,
    )
    logger.info("Split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # Model config
    _FAMILY_ALIASES = {"transformer_light": "transformer", "cnn_bi_lstm": "cnn_bilstm"}
    model_family = _FAMILY_ALIASES.get(
        config.get("model", {}).get("family", "mlp"),
        config.get("model", {}).get("family", "mlp"),
    )
    training_cfg = config.get("training", {})
    _max_epochs  = max_epochs or training_cfg.get("epochs", 30)
    _patience    = training_cfg.get("early_stopping", {}).get("patience", 5)
    _n_trials    = n_trials or 50
    _batch_size  = training_cfg.get("batch_size", 2048)
    window_size  = config.get("data", {}).get("temporal_sequence", {}).get("window_size", 16)

    feature_cols = get_feature_cols(df, label_col)
    n_features   = len(feature_cols[0])
    _num_workers = _compute_num_workers()
    logger.info("Model: %s | n_features=%d | dataloader_workers=%d", model_family, n_features, _num_workers)

    # Build inputs
    is_sequential = model_family in ("cnn_bilstm", "transformer")
    if is_sequential:
        logger.info("Building sliding-window DataLoaders (window_size=%d)…", window_size)
        train_loader, val_loader, test_loader, n_features, preprocessor = build_sequence_inputs(
            train_df, val_df, test_df, feature_cols, label_col,
            window_size=window_size, batch_size=_batch_size, device=device,
            num_workers=_num_workers,
        )
        X_tr = X_vl = X_te = y_tr = y_vl = y_te = None
        train_input, val_input = train_loader, val_loader
    else:
        logger.info("Building tabular DataLoaders…")
        _tab = build_tabular_inputs(
            train_df, val_df, test_df, feature_cols, label_col,
            batch_size=_batch_size, device=device, num_workers=_num_workers,
        )
        (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = _tab[0], _tab[1], _tab[2]
        train_loader, val_loader, test_loader = _tab[3], _tab[4], _tab[5]
        train_input = (X_tr, y_tr) if model_family == "xgboost" else train_loader
        val_input   = (X_vl, y_vl) if model_family == "xgboost" else val_loader

    # Optuna search
    storage_dir = Path(output_dir or f"data/results/{exp_name}")
    storage_dir.mkdir(parents=True, exist_ok=True)
    optuna_storage = f"sqlite:///{storage_dir}/optuna.db"

    logger.info("Running Optuna search (%d trials)…", _n_trials)
    search_result = run_optuna_search(
        model_type=model_family,
        train_data=train_input,
        val_data=val_input,
        n_features=n_features,
        n_classes=2,
        n_trials=_n_trials,
        max_epochs=_max_epochs,
        patience=_patience,
        study_name=exp_name,
        storage=optuna_storage,
        seed=seed,
        device=device,
        use_class_weights=True,
    )
    best_params = search_result["best_params"]
    logger.info("Best val F1: %.4f | params: %s", search_result["best_value"], best_params)

    # Final evaluation on test set
    logger.info("Evaluating best model on test set…")
    if model_family == "xgboost":
        from lift_nids.models.xgboost_wrapper import XGBoostWrapper
        final_model = XGBoostWrapper(**best_params, device=device)
        final_model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
        y_pred  = final_model.predict(X_te)
        y_proba = final_model.predict_proba(X_te)  # sets final_model.infer_time_s
        y_true  = y_te
        efficiency = build_efficiency_record(
            n_params=final_model.n_params,
            train_time_s=final_model.train_time_s,
            infer_time_s=final_model.infer_time_s,
            n_train_samples=len(X_tr),
            n_infer_samples=len(X_te),
            peak_gpu_mem_bytes=0,
            device=device,
        )

    else:
        import torch.nn as nn

        from lift_nids.training.early_stopping import EarlyStopping
        from lift_nids.training.trainer import Trainer, build_optimizer, compute_class_weights

        final_model, lr, wd, _ = build_torch_model(
            model_family, best_params, n_features, 2
        )
        final_model = final_model.to(device)

        label_arr = np.concatenate([y.cpu().numpy() for _, y in train_loader])
        weights   = compute_class_weights(label_arr, 2).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = build_optimizer(final_model, "adam", lr=lr, weight_decay=wd)
        es        = EarlyStopping(patience=_patience, restore_best=True)
        trainer   = Trainer(final_model, optimizer, criterion, device=device, early_stopping=es)
        history   = trainer.fit(train_loader, val_loader, max_epochs=_max_epochs)

        y_true, y_pred, y_proba, _infer_s = measure_pytorch_inference(
            predict_pytorch, final_model, test_loader, device
        )

        efficiency = build_efficiency_record(
            n_params=count_trainable_params(final_model),
            train_time_s=history["train_time_s"],
            infer_time_s=_infer_s,
            n_train_samples=len(train_df),
            n_infer_samples=len(y_true),
            peak_gpu_mem_bytes=history["peak_gpu_mem_bytes"],
            device=device,
        )

    test_metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
    log_experiment_result(logger, exp_name, test_metrics, best_params)

    # Save results
    elapsed = time.time() - t0
    results = {
        "experiment_name": exp_name,
        "model_family":    model_family,
        "seed":            seed,
        "sample_frac":     sample_frac,
        "n_trials":        _n_trials,
        "n_train":         len(train_df),
        "n_val":           len(val_df),
        "n_test":          len(test_df),
        "n_features":      n_features,
        "best_val_f1":     search_result["best_value"],
        "best_params":     best_params,
        "efficiency":      efficiency,
        "elapsed_seconds": round(elapsed, 1),
        **provenance(),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    write_result(results, storage_dir / "results.json", registry_dir=Path("data/results"))
    logger.info("Done in %.1fs → %s", elapsed, storage_dir / "results.json")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = ArgumentParser(description="Run one LiFT-NIDS experiment.")
    parser.add_argument("--config",      default=None)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Subsample fraction for fast runs (e.g. 0.05).")
    parser.add_argument("--n-trials",    type=int,   default=None)
    parser.add_argument("--max-epochs",  type=int,   default=None)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--gpu-check",   action="store_true",
                        help="Run GPU diagnostics and exit without loading datasets.")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Exit with code 1 if any GPU diagnostic check fails.")
    parser.add_argument("--skip-xgboost-gpu-check", action="store_true",
                        help="Skip the optional XGBoost CUDA smoke test.")
    args = parser.parse_args()

    if args.gpu_check:
        report = run_gpu_diagnostics(
            device=args.device or "cuda",
            include_xgboost=not args.skip_xgboost_gpu_check,
        )
        print(json.dumps(report, indent=2))
        if args.require_gpu and not report["ok"]:
            raise SystemExit(1)
        return

    if args.config is None:
        parser.error("--config is required unless --gpu-check is used.")

    config = load_config(args.config)
    run_experiment(
        config=config,
        seed=args.seed,
        sample_frac=args.sample_frac,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        device_override=args.device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
