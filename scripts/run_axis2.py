#!/usr/bin/env python3
"""Axis 2: MAWIFlow forward-chaining runner for publishable temporal results.

Current protocol:
  1. Load MAWIFlow yearly Parquet files and derive the ``year`` column from
     each file stem.
  2. Run one Optuna HPO search per model on the anchor year (``years[0]``)
     using the internal temporal 60/20/20 split: fit / validation /
     internal_test.
  3. Fix the best hyperparameters from that single HPO search and reuse them
     for every forward-chaining window, horizon, and seed.
  4. Evaluate all requested horizons ``k in {1, 2, 3, cumulative}`` with
     R=5 seeds: ``42, 123, 456, 789, 1024``.
  5. For each seed, compute per-window F1+, the median_trajectory over
     delta-t, and nAUT_H as nAUT_1 / nAUT_3 / nAUT_5.
  6. Bootstrap 95% CIs (B=1000) across seeds for nAUT and mean classification
     metrics.
  7. Save per-combination results to
     ``data/results/axis2/<model>/k_<horizon>/results.json`` and the global
     summary to ``data/results/axis2/summary.json``.

CLI arguments:
  --data-path, --output-dir, --models, --horizons, --seeds, --sample-frac,
  --n-trials, --max-epochs, --device, --dry-run,
  --save-checkpoints, --save-attention-artifacts, --attention-samples-per-group,
  --num-workers, --xgboost-device,
  --resume, --skip-existing, --fail-fast.

Production run (one model at a time, Windows-safe):
  uv run python scripts/run_axis2.py --models xgboost --output-dir data/results/axis2_final --num-workers 0 --resume --fail-fast
  uv run python scripts/run_axis2.py --models mlp     --output-dir data/results/axis2_final --num-workers 0 --resume --fail-fast
  uv run python scripts/run_axis2.py --models cnn_bilstm --output-dir data/results/axis2_final --num-workers 0 --resume --fail-fast
  uv run python scripts/run_axis2.py --models transformer --output-dir data/results/axis2_final --num-workers 0 --resume --fail-fast --save-checkpoints --save-attention-artifacts
  uv run python scripts/aggregate_axis2_summaries.py --output-dir data/results/axis2_final

Examples:
  Complete default matrix:
      uv run python scripts/run_axis2.py

  Quick sampled run with fewer HPO trials:
      uv run python scripts/run_axis2.py --sample-frac 0.05 --n-trials 10

  Partial matrix:
      uv run python scripts/run_axis2.py --models transformer xgboost --horizons 1 cumulative

  Pilot run with a single seed (faster iteration, not publishable):
      uv run python scripts/run_axis2.py --seeds 42 --output-dir data/results/axis2_pilot

  Inspect the execution matrix without training:
      uv run python scripts/run_axis2.py --dry-run

  Generate interpretability artifacts for the Transformer (attention maps + checkpoints):
      uv run python scripts/run_axis2.py --models transformer --horizons 1 cumulative \
          --save-checkpoints --save-attention-artifacts
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_experiment import (
    _compute_num_workers,
    _resolve_timestamp_col,
    build_sequence_inputs,
    build_tabular_arrays,
    build_tabular_inputs,
    get_feature_cols,
    load_dataset,
    predict_pytorch,
    resolve_device,
)

from lift_nids.data.dataloader import build_dataloader
from lift_nids.data.dataset import FlowDataset, LazySequenceFlowDataset
from lift_nids.data.preprocessing import FlowPreprocessor
from lift_nids.data.splits import (
    iter_forward_chaining_windows,
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
from lift_nids.experiments.aggregation import aggregate_metrics
from lift_nids.experiments.constants import (
    ALL_MODELS,
    AXIS2_CLASS_METRIC_KEYS,
    AXIS2_TEMPORAL_METRIC_KEYS,
    SEEDS,
)
from lift_nids.experiments.results import build_axis2_record, write_result
from lift_nids.interpretability.attention_maps import plot_attention_heatmap
from lift_nids.models.factory import build_torch_model
from lift_nids.models.xgboost_wrapper import XGBoostWrapper
from lift_nids.training.early_stopping import EarlyStopping
from lift_nids.training.optuna_search import run_optuna_search
from lift_nids.training.trainer import Trainer, build_optimizer, compute_class_weights
from lift_nids.utils.io import save_results
from lift_nids.utils.logging import setup_logger
from lift_nids.utils.reproducibility import set_global_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_HORIZONS: list[int | str] = [1, 2, 3, "cumulative"]

_MAWIFLOW_CFG: dict[str, Any] = {    "dataset": {
        "name": "MAWIFlow",
        "path": "data/processed/mawiflow",
        "file_format": "parquet",
        "label_column": "label",
        "timestamp_column": "timestamp",
    },
    "data": {
        "temporal_sequence": {"window_size": 16},
    },
    "training": {
        "device": "auto",
        "epochs": 30,
        "batch_size": 2048,
        "early_stopping": {"patience": 5},
    },
}


# ---------------------------------------------------------------------------
# WindowInputContext — preprocessor fitted once per forward-chaining window
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WindowInputContext:
    """Per-window preprocessing context: FlowPreprocessor fitted once on fit_w_df.

    Holds pre-built train/val data and provides helpers to transform any test
    split using the already-fitted preprocessor — eliminating repeated fit_transform
    calls within the same forward-chaining window.

    This is an operational optimisation; it does not change the experimental
    protocol because all transforms use the same fit_w_df-fitted statistics.
    """

    preprocessor: FlowPreprocessor
    feat_cols: list[str]    # column names after preprocessing
    label_col: str
    window_size: int        # 0 for tabular models
    n_features: int

    # Pre-built train / val data (set by builder functions below)
    X_train: np.ndarray | None = dataclasses.field(default=None, repr=False)
    y_train: np.ndarray | None = dataclasses.field(default=None, repr=False)
    X_val: np.ndarray | None = dataclasses.field(default=None, repr=False)
    y_val: np.ndarray | None = dataclasses.field(default=None, repr=False)
    train_loader: Any | None = dataclasses.field(default=None, repr=False)
    val_loader: Any | None = dataclasses.field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Test-split helpers — never call fit() or fit_transform()
    # ------------------------------------------------------------------

    def make_test_arrays(
        self, test_df: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform test_df with the fitted preprocessor (tabular path)."""
        X = self.preprocessor.transform(test_df).to_numpy(np.float32)
        y = test_df[self.label_col].to_numpy(np.int64)
        return X, y

    def make_test_tabular_loader(
        self,
        test_df: Any,
        batch_size: int,
        device: str,
        num_workers: int,
    ) -> Any:
        """Build a DataLoader for tabular test data without refitting."""
        X, y = self.make_test_arrays(test_df)
        pin = "cuda" in device
        return build_dataloader(
            FlowDataset(X, y), batch_size=batch_size,
            pin_memory=pin, num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    def make_test_sequence_loader(
        self,
        test_df: Any,
        batch_size: int,
        device: str,
        num_workers: int,
    ) -> Any:
        """Build a sequence DataLoader for test data without refitting."""
        X_test_df = self.preprocessor.transform(test_df)
        X_arr = X_test_df.to_numpy(np.float32)
        y_arr = test_df[self.label_col].reset_index(drop=True).to_numpy(np.int64)
        pin = "cuda" in device
        return build_dataloader(
            LazySequenceFlowDataset(X_arr, y_arr, self.window_size, stride=1),
            batch_size=batch_size,
            pin_memory=pin, num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )


def _build_sequential_window_context(
    fit_w_df: Any,
    val_w_df: Any,
    feature_cols: tuple[list[str], list[str]],
    label_col: str,
    window_size: int,
    batch_size: int,
    device: str,
    num_workers: int,
) -> WindowInputContext:
    """Fit preprocessor once on fit_w_df; build train/val sequence loaders."""
    num_cols, cat_cols = feature_cols
    preprocessor = FlowPreprocessor(log_transform=True)
    X_train_df = preprocessor.fit_transform(fit_w_df, num_cols, cat_cols)
    X_val_df = preprocessor.transform(val_w_df)

    feat_cols = list(X_train_df.columns)
    X_train_arr = X_train_df.to_numpy(np.float32)
    y_train_arr = fit_w_df[label_col].reset_index(drop=True).to_numpy(np.int64)
    X_val_arr = X_val_df.to_numpy(np.float32)
    y_val_arr = val_w_df[label_col].reset_index(drop=True).to_numpy(np.int64)

    pin = "cuda" in device

    persistent = num_workers > 0
    return WindowInputContext(
        preprocessor=preprocessor,
        feat_cols=feat_cols,
        label_col=label_col,
        window_size=window_size,
        n_features=X_train_arr.shape[1],
        y_train=y_train_arr,
        train_loader=build_dataloader(
            LazySequenceFlowDataset(X_train_arr, y_train_arr, window_size, stride=window_size // 2),
            batch_size=batch_size, shuffle=True,
            pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
        ),
        val_loader=build_dataloader(
            LazySequenceFlowDataset(X_val_arr, y_val_arr, window_size, stride=1),
            batch_size=batch_size,
            pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
        ),
    )


def _build_tabular_window_context(
    fit_w_df: Any,
    val_w_df: Any,
    feature_cols: tuple[list[str], list[str]],
    label_col: str,
    batch_size: int | None = None,
    device: str = "cpu",
    num_workers: int = 0,
    build_loaders: bool = True,
) -> WindowInputContext:
    """Fit preprocessor once on fit_w_df; build train/val arrays (and optionally loaders)."""
    num_cols, cat_cols = feature_cols
    preprocessor = FlowPreprocessor(log_transform=True)
    X_train_df = preprocessor.fit_transform(fit_w_df, num_cols, cat_cols)
    feat_cols = list(X_train_df.columns)
    X_train = X_train_df.to_numpy(np.float32)
    X_val = preprocessor.transform(val_w_df).to_numpy(np.float32)
    y_train = fit_w_df[label_col].to_numpy(np.int64)
    y_val = val_w_df[label_col].to_numpy(np.int64)

    train_loader = val_loader = None
    if build_loaders and batch_size is not None:
        pin = "cuda" in device
        persistent = num_workers > 0
        train_loader = build_dataloader(
            FlowDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
            pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
        )
        val_loader = build_dataloader(
            FlowDataset(X_val, y_val), batch_size=batch_size,
            pin_memory=pin, num_workers=num_workers, persistent_workers=persistent,
        )

    return WindowInputContext(
        preprocessor=preprocessor,
        feat_cols=feat_cols,
        label_col=label_col,
        window_size=0,
        n_features=X_train.shape[1],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_loader=train_loader,
        val_loader=val_loader,
    )


# ---------------------------------------------------------------------------
# Resume / skip compatibility check
# ---------------------------------------------------------------------------

def _frac_key(frac: float | None) -> str:
    """Normalise sample_frac: None / ≥1.0 → 'full', else str of rounded value."""
    if frac is None or frac >= 1.0:
        return "full"
    return str(round(frac, 6))


def _check_results_compatible(
    existing: dict,
    model_family: str,
    horizon: int | str,
    seeds: list[int],
    n_trials: int,
    sample_frac: float | None,
    run_mode: str,
) -> tuple[bool, str]:
    """Return (is_compatible, reason_if_not) for an existing results.json.

    Checks that the stored run configuration matches the current one so that
    smoke-test (--sample-frac) and production results are never silently mixed.
    """
    scalar_checks: list[tuple[str, Any, Any]] = [
        ("model_family", existing.get("model_family"), model_family),
        ("dataset",      existing.get("dataset"),      "MAWIFlow"),
        ("axis",         existing.get("axis"),          2),
        ("horizon",      str(existing.get("horizon")), str(horizon)),
        ("n_trials",     existing.get("n_trials"),     n_trials),
        ("seeds",        existing.get("seeds"),         seeds),
    ]
    for field, got, want in scalar_checks:
        if got != want:
            return False, f"{field} mismatch: existing={got!r}, current={want!r}"

    # sample_frac: treat None and ≥1.0 as equivalent ("full run")
    existing_frac_key = _frac_key(existing.get("sample_frac"))
    current_frac_key = _frac_key(sample_frac)
    if existing_frac_key != current_frac_key:
        return False, (
            f"sample_frac mismatch: existing={existing.get('sample_frac')!r} "
            f"(→{existing_frac_key}), current={sample_frac!r} (→{current_frac_key})"
        )

    # run_mode (production vs pilot): only check when the existing result recorded it
    existing_mode = existing.get("run_mode")
    if existing_mode is not None and existing_mode != run_mode:
        return False, f"run_mode mismatch: existing={existing_mode!r}, current={run_mode!r}"

    return True, ""


def _load_existing_horizon_result(
    model_dir: Path,
    model_family: str,
    horizon: int | str,
    seeds: list[int],
    n_trials: int,
    sample_frac: float | None,
    run_mode: str,
    skip_existing: bool,
) -> tuple[str, dict | None]:
    """Check one horizon's results.json before data loading and HPO.

    Returns:
        ("compatible", existing_dict) — result exists, valid, and passes compatibility.
            existing_dict already has skipped_existing=True.
        ("incompatible", None) — result exists but fails compatibility; caller
            re-runs (only reached when resume=True, not skip_existing=True).
        ("missing", None) — no result, unreadable file, or result with error key.

    Raises:
        RuntimeError: when skip_existing=True and an incompatible result is found.
    """
    h_key = _horizon_dir_name(horizon)
    results_path = model_dir / h_key / "results.json"
    if not results_path.exists():
        return "missing", None
    try:
        existing = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "missing", None
    if "error" in existing:
        return "missing", None
    is_compat, reason = _check_results_compatible(
        existing, model_family, horizon, seeds, n_trials, sample_frac, run_mode
    )
    if is_compat:
        existing["skipped_existing"] = True
        return "compatible", existing
    if skip_existing:
        raise RuntimeError(
            f"--skip-existing: incompatible existing result for "
            f"{model_family}/{h_key}: {reason}. "
            "Remove the file to re-run."
        )
    return "incompatible", None


# ---------------------------------------------------------------------------
# Transformer interpretability helpers
# ---------------------------------------------------------------------------

def _save_transformer_checkpoint(
    model: torch.nn.Module,
    window_dir: Path,
    metadata: dict,
) -> Path:
    """Save state_dict and metadata JSON for one transformer forward-chaining window."""
    window_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = window_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)
    with open(window_dir / "checkpoint_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return ckpt_path


def _collect_attention_by_class(
    model: torch.nn.Module,
    loader: Any,
    device: str,
    n_per_group: int,
) -> dict:
    """Collect per-class averaged last-layer attention from correctly classified samples.

    Iterates ``loader`` until ``n_per_group`` correctly classified attack (label=1)
    and benign (label=0) samples are found.  Returns averaged attention matrices
    ``(n_heads, seq_len, seq_len)`` per class, or ``None`` when a class has no
    correctly classified samples.

    Args:
        model:        Transformer in eval mode with ``return_attention_weights`` support.
        loader:       DataLoader yielding ``(X, y)`` batches.
        device:       Device string.
        n_per_group:  Maximum samples to collect per class.

    Returns:
        Dict with keys ``attack``, ``benign`` (ndarray or None),
        ``n_attack``, ``n_benign`` (int).
    """
    attack_attns: list[np.ndarray] = []
    benign_attns: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for X, y in loader:
            if len(attack_attns) >= n_per_group and len(benign_attns) >= n_per_group:
                break
            logits, attn_list = model(X.to(device), return_attention_weights=True)
            preds = logits.argmax(dim=1).cpu()
            attn_last = attn_list[-1].cpu().numpy()  # (batch, n_heads, seq_len, seq_len)
            for i in range(len(y)):
                label = int(y[i])
                pred = int(preds[i])
                if pred != label:
                    continue
                if label == 1 and len(attack_attns) < n_per_group:
                    attack_attns.append(attn_last[i])  # (n_heads, seq_len, seq_len)
                elif label == 0 and len(benign_attns) < n_per_group:
                    benign_attns.append(attn_last[i])

    return {
        "attack": np.mean(attack_attns, axis=0) if attack_attns else None,
        "benign": np.mean(benign_attns, axis=0) if benign_attns else None,
        "n_attack": len(attack_attns),
        "n_benign": len(benign_attns),
    }


def _generate_attention_artifacts(
    model: torch.nn.Module,
    loader_delta0: Any,
    loader_future: Any | None,
    device: str,
    artifacts_dir: Path,
    n_per_group: int = 16,
    delta_t_future: int | None = None,
) -> dict:
    """Generate attention heatmaps and summary JSON for one transformer window.

    Plots averaged attention (mean over heads) for correctly classified attack/
    benign samples at Δt=0 (internal test) and at Δt>0 (first future year).
    Gracefully records a warning when a group has no correctly classified samples.

    Args:
        model:          Trained LightTransformer in eval mode.
        loader_delta0:  Internal test DataLoader (Δt=0).
        loader_future:  First future year DataLoader (Δt>0), or None.
        device:         Device string.
        artifacts_dir:  Directory where PNGs and ``attention_summary.json`` are saved.
        n_per_group:    Maximum samples per class per scenario.
        delta_t_future: Δt in years for the future loader (for metadata).

    Returns:
        Summary dict mirroring ``attention_summary.json``.
    """
    import matplotlib
    matplotlib.use("Agg")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    png_paths: list[str] = []

    def _plot_avg(avg_attn: np.ndarray | None, png_path: Path, label: str) -> None:
        if avg_attn is None:
            warnings_list.append(f"No correctly classified {label} samples — skipping heatmap.")
            return
        # avg_attn: (n_heads, seq_len, seq_len) → mean over heads → (seq_len, seq_len)
        plot_attention_heatmap(avg_attn.mean(axis=0), output_path=png_path)
        png_paths.append(str(png_path))

    # Δt=0
    d0 = _collect_attention_by_class(model, loader_delta0, device, n_per_group)
    _plot_avg(d0["attack"], artifacts_dir / "attention_attack_delta0.png", "attack (Δt=0)")
    _plot_avg(d0["benign"], artifacts_dir / "attention_benign_delta0.png", "benign (Δt=0)")

    # Δt>0 (first future year)
    fut: dict = {"n_attack": 0, "n_benign": 0}
    if loader_future is not None:
        fut = _collect_attention_by_class(model, loader_future, device, n_per_group)
        _plot_avg(fut["attack"], artifacts_dir / "attention_attack_future.png", "attack (future)")
        _plot_avg(fut["benign"], artifacts_dir / "attention_benign_future.png", "benign (future)")
    else:
        warnings_list.append("No future year loader available — Δt>0 heatmaps not generated.")

    summary = {
        "layer_used":             -1,
        "heads_aggregated":       "mean_over_all_heads",
        "n_per_group_requested":  n_per_group,
        "delta0": {
            "n_attack": d0["n_attack"],
            "n_benign": d0["n_benign"],
        },
        "future": {
            "delta_t":  delta_t_future,
            "n_attack": fut.get("n_attack", 0),
            "n_benign": fut.get("n_benign", 0),
        },
        "png_paths": png_paths,
        "warnings":  warnings_list,
    }
    with open(artifacts_dir / "attention_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_horizons(raw: list[str]) -> list[int | str]:
    """Convert CLI horizon strings to typed values ('cumulative' stays; others become int)."""
    result: list[int | str] = []
    for h in raw:
        if h == "cumulative":
            result.append("cumulative")
        else:
            result.append(int(h))
    return result


def _horizon_dir_name(horizon: int | str) -> str:
    """Return the output subdirectory name for a given horizon (e.g. 'k_1', 'k_cumulative')."""
    return f"k_{horizon}"


def _run_forward_chaining_seed(
    model_family: str,
    best_params: dict,
    n_features: int,
    df: Any,
    year_col: str,
    label_col: str,
    feature_cols: tuple,
    window_size: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
    seed: int,
    horizon: int | str,
    num_workers: int,
    xgb_device: str = "cpu",
    timestamp_col: str | None = None,
    save_checkpoints: bool = False,
    save_attention_artifacts: bool = False,
    attention_samples_per_group: int = 16,
    seed_artifacts_dir: Path | None = None,
) -> dict:
    """One complete forward-chaining run for a single seed.

    The preprocessor is fitted exactly once per forward-chaining window on
    fit_w_df, then reused (transform-only) for all test splits within that
    window.  This eliminates repeated fit_transform calls without changing
    the experimental protocol.

    Returns per-window metrics, nAUT_H, trajectory, and mean classification
    metrics aggregated across all forward-chaining windows.
    """
    is_sequential = model_family in ("cnn_bilstm", "transformer")
    splits = iter_forward_chaining_windows(df, year_col=year_col, k=horizon, is_sorted=True)

    years_fc = sorted(int(y) for y in df[year_col].unique())
    n_windows = (
        len(years_fc) - 1
        if horizon == "cumulative"
        else max(0, len(years_fc) - int(horizon))
    )

    f1plus_records: list[tuple[int, int, float]] = []
    window_results: list[dict] = []
    window_cls_metrics: list[dict] = []
    window_efficiency: list[dict] = []

    pbar = tqdm(
        enumerate(splits),
        total=n_windows,
        desc=f"{model_family} k={horizon} seed={seed}",
        unit="win",
        dynamic_ncols=True,
        leave=True,
    )
    for win_idx, win in pbar:
        # 60/20/20 temporal split: fit / val / internal_test (for Δt=0 evaluation)
        fit_w_df, val_w_df, internal_test_df = temporal_train_val_test_split(
            win.train_df, year_col, timestamp_col=timestamp_col, already_sorted=True
        )

        def _record(test_year: int, metrics: dict, f1_plus: float) -> None:
            f1plus_records.append((win.anchor_year, test_year, f1_plus))
            window_results.append({
                "window":      win_idx,
                "train_years": win.train_years,
                "anchor_year": win.anchor_year,
                "test_year":   test_year,
                **{f"test_{mk}": mv for mk, mv in metrics.items()},
            })
            window_cls_metrics.append(metrics)

        if is_sequential:
            # Build context: fit preprocessor ONCE on fit_w_df
            ctx = _build_sequential_window_context(
                fit_w_df, val_w_df, feature_cols, label_col,
                window_size=window_size, batch_size=batch_size,
                device=device, num_workers=num_workers,
            )

            final_model, lr, wd, _ = build_torch_model(
                model_family, best_params, ctx.n_features, 2
            )
            final_model = final_model.to(device)
            label_arr = ctx.y_train
            weights = compute_class_weights(label_arr, 2).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = build_optimizer(final_model, "adam", lr=lr, weight_decay=wd)
            es = EarlyStopping(patience=patience, restore_best=True)
            _hist_w = Trainer(final_model, optimizer, criterion, device=device,
                              early_stopping=es, log_train_metrics=False).fit(
                ctx.train_loader, ctx.val_loader, max_epochs=max_epochs
            )

            # Δt=0: internal test slice — reuse fitted preprocessor from context
            int_tst_l = ctx.make_test_sequence_loader(
                internal_test_df, batch_size, device, num_workers
            )
            y_true_int, y_pred_int, y_proba_int, _infer_w_s = measure_pytorch_inference(
                predict_pytorch, final_model, int_tst_l, device
            )
            window_efficiency.append(build_efficiency_record(
                n_params=count_trainable_params(final_model),
                train_time_s=_hist_w["train_time_s"],
                infer_time_s=_infer_w_s,
                n_train_samples=len(fit_w_df),
                n_infer_samples=len(y_true_int),
                peak_gpu_mem_bytes=_hist_w["peak_gpu_mem_bytes"],
                device=device,
            ))
            metrics_int = compute_classification_metrics(
                y_true_int, y_pred_int, y_proba=y_proba_int
            )
            _record(win.anchor_year, metrics_int, _extract_f1_plus(metrics_int))

            # -- Transformer-only: checkpoints and attention artifacts --------
            if model_family == "transformer" and seed_artifacts_dir is not None:
                win_dir = seed_artifacts_dir / f"window_{win_idx}"
                if save_checkpoints:
                    _save_transformer_checkpoint(final_model, win_dir, {
                        "seed":        seed,
                        "horizon":     str(horizon),
                        "window_idx":  win_idx,
                        "train_years": win.train_years,
                        "anchor_year": win.anchor_year,
                        "window_size": window_size,
                        "n_features":  ctx.n_features,
                        "best_params": best_params,
                    })
                if save_attention_artifacts:
                    first_future_yr = sorted(win.test_dfs_by_year.keys())[0]
                    fut_df = win.test_dfs_by_year[first_future_yr].reset_index(drop=True)
                    # Reuse context: no new fit_transform
                    te_attn_l = ctx.make_test_sequence_loader(
                        fut_df, batch_size, device, num_workers
                    )
                    _generate_attention_artifacts(
                        model=final_model,
                        loader_delta0=int_tst_l,
                        loader_future=te_attn_l,
                        device=device,
                        artifacts_dir=win_dir / "attention",
                        n_per_group=attention_samples_per_group,
                        delta_t_future=first_future_yr - win.anchor_year,
                    )

            for test_yr, test_yr_raw_df in sorted(win.test_dfs_by_year.items()):
                tst_yr_df = test_yr_raw_df.reset_index(drop=True)
                # Reuse context: no new fit_transform for each future year
                te_yr_l = ctx.make_test_sequence_loader(
                    tst_yr_df, batch_size, device, num_workers
                )
                y_true_yr, y_pred_yr, y_proba_yr = predict_pytorch(
                    final_model, te_yr_l, device
                )
                metrics_yr = compute_classification_metrics(
                    y_true_yr, y_pred_yr, y_proba=y_proba_yr
                )
                _record(test_yr, metrics_yr, _extract_f1_plus(metrics_yr))
            del final_model, optimizer, weights, criterion, es, int_tst_l, label_arr

        elif model_family == "xgboost":
            # ── XGBoost: numpy-only path ──────────────────────────────────
            # Build context: fit preprocessor ONCE on fit_w_df
            ctx = _build_tabular_window_context(
                fit_w_df, val_w_df, feature_cols, label_col,
                build_loaders=False,
            )
            X_te_int, y_te_int = ctx.make_test_arrays(internal_test_df)

            model_w = XGBoostWrapper(**best_params, seed=seed, device=xgb_device)
            model_w.fit(ctx.X_train, ctx.y_train, X_val=ctx.X_val, y_val=ctx.y_val)

            y_pred_int  = model_w.predict(X_te_int)
            y_proba_int = model_w.predict_proba(X_te_int)
            window_efficiency.append(build_efficiency_record(
                n_params=model_w.n_params,
                train_time_s=model_w.train_time_s,
                infer_time_s=model_w.infer_time_s,
                n_train_samples=len(ctx.X_train),
                n_infer_samples=len(y_te_int),
                peak_gpu_mem_bytes=0,
                device=xgb_device,
            ))
            metrics_int = compute_classification_metrics(
                y_te_int, y_pred_int, y_proba=y_proba_int
            )
            _record(win.anchor_year, metrics_int, _extract_f1_plus(metrics_int))

            for test_yr, test_yr_raw_df in sorted(win.test_dfs_by_year.items()):
                tst_yr_df = test_yr_raw_df.reset_index(drop=True)
                # Reuse context: no new fit_transform for each future year
                X_te_yr, y_te_yr = ctx.make_test_arrays(tst_yr_df)
                y_pred_yr  = model_w.predict(X_te_yr)
                y_proba_yr = model_w.predict_proba(X_te_yr)
                metrics_yr = compute_classification_metrics(
                    y_te_yr, y_pred_yr, y_proba=y_proba_yr
                )
                _record(test_yr, metrics_yr, _extract_f1_plus(metrics_yr))
            del model_w, X_te_int, y_te_int

        else:
            # ── MLP: keeps DataLoader path ────────────────────────────────
            # Build context: fit preprocessor ONCE on fit_w_df
            ctx = _build_tabular_window_context(
                fit_w_df, val_w_df, feature_cols, label_col,
                batch_size=batch_size, device=device, num_workers=num_workers,
            )

            final_model, lr, wd, _ = build_torch_model(
                model_family, best_params, ctx.n_features, 2
            )
            final_model = final_model.to(device)
            label_arr = ctx.y_train
            weights = compute_class_weights(label_arr, 2).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = build_optimizer(final_model, "adam", lr=lr, weight_decay=wd)
            es = EarlyStopping(patience=patience, restore_best=True)
            _hist_tab = Trainer(final_model, optimizer, criterion, device=device,
                                early_stopping=es, log_train_metrics=False).fit(
                ctx.train_loader, ctx.val_loader, max_epochs=max_epochs
            )

            # Δt=0: internal test slice — reuse fitted preprocessor from context
            te_int_l = ctx.make_test_tabular_loader(
                internal_test_df, batch_size, device, num_workers
            )
            y_true_int, y_pred_int, y_proba_int, _infer_tab_s = measure_pytorch_inference(
                predict_pytorch, final_model, te_int_l, device
            )
            window_efficiency.append(build_efficiency_record(
                n_params=count_trainable_params(final_model),
                train_time_s=_hist_tab["train_time_s"],
                infer_time_s=_infer_tab_s,
                n_train_samples=len(ctx.X_train),
                n_infer_samples=len(y_true_int),
                peak_gpu_mem_bytes=_hist_tab["peak_gpu_mem_bytes"],
                device=device,
            ))
            metrics_int = compute_classification_metrics(
                y_true_int, y_pred_int, y_proba=y_proba_int
            )
            _record(win.anchor_year, metrics_int, _extract_f1_plus(metrics_int))

            for test_yr, test_yr_raw_df in sorted(win.test_dfs_by_year.items()):
                tst_yr_df = test_yr_raw_df.reset_index(drop=True)
                # Reuse context: no new fit_transform for each future year
                te_yr_l = ctx.make_test_tabular_loader(
                    tst_yr_df, batch_size, device, num_workers
                )
                y_true_yr, y_pred_yr, y_proba_yr = predict_pytorch(
                    final_model, te_yr_l, device
                )
                metrics_yr = compute_classification_metrics(
                    y_true_yr, y_pred_yr, y_proba=y_proba_yr
                )
                _record(test_yr, metrics_yr, _extract_f1_plus(metrics_yr))
            del final_model, optimizer, weights, criterion, es, te_int_l, label_arr

        # Update progress bar: anchor year + Δt=0 F1+ (first entry for this window)
        n_future = len(win.test_dfs_by_year)
        if window_cls_metrics and len(window_cls_metrics) >= n_future + 1:
            delta0_f1p = _extract_f1_plus(window_cls_metrics[-(n_future + 1)])
            pbar.set_postfix({"anchor": win.anchor_year, "F1+(Δt=0)": f"{delta0_f1p:.3f}"})

        # Release this window's data before moving to the next
        del fit_w_df, val_w_df, internal_test_df, win, ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    traj = median_trajectory(f1plus_records)
    naut_1 = compute_naut(traj, H=1)
    naut_3 = compute_naut(traj, H=3)
    naut_5 = compute_naut(traj, H=5)

    # Mean classification metrics across all forward-chaining windows
    mean_cls: dict[str, float] = {}
    for mk in AXIS2_CLASS_METRIC_KEYS:
        vals = [m.get(mk, float("nan")) for m in window_cls_metrics]
        valid = [v for v in vals if not np.isnan(v)]
        mean_cls[mk] = float(np.mean(valid)) if valid else float("nan")

    return {
        "seed":             seed,
        "nAUT_1":           naut_1,
        "nAUT_3":           naut_3,
        "nAUT_5":           naut_5,
        "median_trajectory": traj,
        "window_results":   window_results,
        "f1plus_records":   f1plus_records,
        "window_efficiency": window_efficiency,
        **mean_cls,
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_family: str,
    config: dict,
    sample_frac: float | None,
    n_trials: int,
    max_epochs: int,
    horizons: list[int | str],
    device: str,
    output_dir: Path,
    logger: Any,
    seeds: list[int] | None = None,
    save_checkpoints: bool = False,
    save_attention_artifacts: bool = False,
    attention_samples_per_group: int = 16,
    num_workers: int = 0,
    xgb_device: str = "cpu",
    resume: bool = False,
    skip_existing: bool = False,
    fail_fast: bool = False,
    run_mode: str = "production",
) -> dict[str, dict]:
    """Full Axis 2 pipeline for one model: HPO once + R forward-chaining seeds per horizon.

    HPO is performed a single time on the anchor year (years[0]) and the resulting
    best_params are reused across ALL horizons — consistent with the single-HPO policy.

    When ``resume`` or ``skip_existing`` is True, an existing ``results.json`` that
    passes compatibility checks is reused without re-running.  Skipped combinations
    have ``"skipped_existing": True`` in the returned dict.

    When ``fail_fast`` is True, the first horizon failure raises immediately.

    Returns a dict keyed by horizon directory name (e.g. 'k_1', 'k_cumulative'),
    where each value is the full result dict for that combination.  On per-horizon
    failure the value is ``{"error": "<message>"}``.  On skip the value has
    ``{"skipped_existing": True, ...existing fields...}``.
    """
    if seeds is None:
        seeds = SEEDS
    label_col = config["dataset"]["label_column"]
    patience = config["training"]["early_stopping"]["patience"]
    batch_size = config["training"]["batch_size"]
    window_size = config["data"]["temporal_sequence"]["window_size"]
    is_sequential = model_family in ("cnn_bilstm", "transformer")

    model_dir = output_dir / model_family  # defined early for pre-check

    logger.info("=== Model: %s | horizons=%s ===", model_family, horizons)

    # -- Early resume/skip check before expensive data load and HPO ----------
    existing_results: dict[str, dict] = {}
    pending_horizons: list[int | str] = list(horizons)

    if resume or skip_existing:
        pending_horizons = []
        for horizon in horizons:
            status, existing = _load_existing_horizon_result(
                model_dir, model_family, horizon, seeds, n_trials, sample_frac,
                run_mode, skip_existing,
            )
            if status == "compatible":
                h_key = _horizon_dir_name(horizon)
                existing_results[h_key] = existing
                logger.info(
                    "  [k=%s] Pre-check: compatible result already exists.",
                    horizon,
                )
            else:
                pending_horizons.append(horizon)

        if not pending_horizons:
            logger.info(
                "All horizons for %s have compatible results — skipping data load and HPO.",
                model_family,
            )
            return existing_results

    # -- Load MAWIFlow -------------------------------------------------------
    logger.info("Loading MAWIFlow (sample_frac=%s)…", sample_frac)
    df = load_dataset(config, sample_frac=sample_frac, seed=seeds[0])
    # Downcast memory-safe columns: year ∈ [2007,2024] fits int16; binary label fits int8.
    year_col      = "year"
    if year_col in df.columns:
        df[year_col] = df[year_col].astype("int16")
    if label_col in df.columns:
        df[label_col] = df[label_col].astype("int8")
    timestamp_col = _resolve_timestamp_col(df, config)
    if timestamp_col is None:
        raise ValueError(
            "MAWIFlow temporal protocol requires an auditable timestamp column for "
            "chronological ordering, but none was found in the loaded DataFrame. "
            f"Config expects '{config['dataset'].get('timestamp_column', 'timestamp')}'. "
            "Ensure the yearly parquet files contain a valid timestamp column."
        )
    if year_col not in df.columns:
        raise ValueError(
            "Column 'year' not found. "
            "Ensure data/processed/mawiflow/ contains yearly *.parquet files."
        )
    years = sorted(df[year_col].unique())
    if len(years) < 2:
        raise ValueError(f"Forward-chaining requires ≥ 2 years; got {years}")
    logger.info("Loaded %d rows | years: %s | timestamp_col=%s", len(df), years, timestamp_col)

    # Sort once by (year, timestamp) so all forward-chaining iterations can use
    # zero-copy iloc slicing instead of re-allocating per-window copies.
    logger.info("Sorting by (%s, %s) for zero-copy forward-chaining…", year_col, timestamp_col)
    df = df.sort_values([year_col, timestamp_col], kind="stable")

    feature_cols = get_feature_cols(df, label_col)

    # -- HPO on year[0] with temporal 60/20/20 split — once per model -------
    anchor_df = df[df[year_col] == years[0]].reset_index(drop=True)
    train_hpo_df, val_hpo_df, _ = temporal_train_val_test_split(
        anchor_df, year_col, timestamp_col=timestamp_col
    )
    logger.info(
        "HPO anchor year=%d: fit=%d  val=%d  (temporal 60/20/20; internal test not used in HPO)",
        years[0], len(train_hpo_df), len(val_hpo_df),
    )

    if is_sequential:
        hpo_tr_l, hpo_vl_l, _, n_features, _ = build_sequence_inputs(
            train_hpo_df, val_hpo_df, val_hpo_df, feature_cols, label_col,
            window_size=window_size, batch_size=batch_size,
            device=device, num_workers=num_workers,
        )
        hpo_train_input, hpo_val_input = hpo_tr_l, hpo_vl_l
    elif model_family == "xgboost":
        # Numpy-only HPO path for XGBoost — no DataLoader workers needed
        (X_htr, y_htr), (X_hvl, y_hvl), _, _ = build_tabular_arrays(
            train_hpo_df, val_hpo_df, val_hpo_df, feature_cols, label_col,
        )
        n_features = X_htr.shape[1]
        hpo_train_input = (X_htr, y_htr)
        hpo_val_input = (X_hvl, y_hvl)
    else:
        _tab_hpo = build_tabular_inputs(
            train_hpo_df, val_hpo_df, val_hpo_df, feature_cols, label_col,
            batch_size=batch_size, device=device, num_workers=num_workers,
        )
        (X_htr, y_htr), (X_hvl, y_hvl) = _tab_hpo[0], _tab_hpo[1]
        n_features = X_htr.shape[1]
        hpo_train_input = _tab_hpo[3]
        hpo_val_input = _tab_hpo[4]

    model_dir.mkdir(parents=True, exist_ok=True)
    optuna_storage = f"sqlite:///{model_dir}/optuna.db"

    hpo_device = xgb_device if model_family == "xgboost" else device
    logger.info(
        "[HPO] %d trials for %s (device=%s)…",
        n_trials,
        model_family,
        hpo_device,
    )
    set_global_seed(seeds[0])
    search_result = run_optuna_search(
        model_type=model_family,
        train_data=hpo_train_input,
        val_data=hpo_val_input,
        n_features=n_features,
        n_classes=2,
        n_trials=n_trials,
        max_epochs=max_epochs,
        patience=patience,
        study_name=f"axis2_{model_family}",
        storage=optuna_storage,
        seed=seeds[0],
        device=hpo_device,
        use_class_weights=True,
    )
    best_params = search_result["best_params"]
    logger.info("[HPO] Best val F1=%.4f  params=%s",
                search_result["best_value"], best_params)

    # For PyTorch models, honour the batch_size chosen by Optuna.
    # XGBoost has no batch_size in its search space — keep the config default.
    if model_family in ("mlp", "cnn_bilstm", "transformer"):
        effective_batch_size = int(best_params.get("batch_size", batch_size))
    else:
        effective_batch_size = batch_size
    if effective_batch_size != batch_size:
        logger.info(
            "[HPO] Overriding config batch_size %d → effective_batch_size %d (from best_params)",
            batch_size, effective_batch_size,
        )

    # -- Loop over pending horizons: HPO params are reused for each --------
    horizon_results: dict[str, dict] = dict(existing_results)
    for horizon in pending_horizons:
        h_key = _horizon_dir_name(horizon)
        horizon_dir = model_dir / h_key
        horizon_dir.mkdir(parents=True, exist_ok=True)
        comb = f"{model_family}/{h_key}"

        # -- Resume / skip-existing check -----------------------------------
        results_path = horizon_dir / "results.json"
        if (resume or skip_existing) and results_path.exists():
            try:
                existing = json.loads(results_path.read_text(encoding="utf-8"))
                if "error" not in existing:
                    is_compat, reason = _check_results_compatible(
                        existing, model_family, horizon, seeds,
                        n_trials, sample_frac, run_mode,
                    )
                    if is_compat:
                        logger.info(
                            "  [k=%s] Skipping %s — compatible result already exists.",
                            horizon, comb,
                        )
                        existing["skipped_existing"] = True
                        horizon_results[h_key] = existing
                        continue
                    else:
                        if skip_existing:
                            raise RuntimeError(
                                f"--skip-existing: incompatible existing result for {comb}: {reason}. "
                                "Remove the file or use --force (not yet implemented) to overwrite."
                            )
                        logger.warning(
                            "  [k=%s] Existing result for %s is incompatible (%s); re-running.",
                            horizon, comb, reason,
                        )
                else:
                    logger.warning(
                        "  [k=%s] Existing result for %s contains error; re-running.",
                        horizon, comb,
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "  [k=%s] Could not read existing result for %s (%s); re-running.",
                    horizon, comb, e,
                )

        logger.info("  [k=%s] Running R=%d seeds…", horizon, len(seeds))
        try:
            t_h = time.time()
            per_seed_results: list[dict] = []
            _seed_resume_keys = ("nAUT_1", "nAUT_3", "nAUT_5") + tuple(AXIS2_CLASS_METRIC_KEYS)
            for seed in seeds:
                seed_json_path = horizon_dir / f"seed_{seed}.json"

                # Seed-level resume: reuse if file has all required metric keys.
                # window_efficiency may be absent in older files; default to [] so
                # aggregation still works (efficiency summary covers available seeds only).
                if resume and seed_json_path.exists():
                    try:
                        _loaded = json.loads(
                            seed_json_path.read_text(encoding="utf-8")
                        )
                        if (
                            "error" not in _loaded
                            and _loaded.get("seed") == seed
                            and all(k in _loaded for k in _seed_resume_keys)
                        ):
                            _loaded.setdefault("window_efficiency", [])
                            _loaded.setdefault("window_results", [])
                            _loaded.setdefault("f1plus_records", [])
                            _loaded.setdefault("median_trajectory", {})
                            logger.info(
                                "    [Seed %d] Reusing existing result "
                                "(nAUT_1=%.4f  nAUT_3=%.4f  nAUT_5=%.4f).",
                                seed,
                                _loaded["nAUT_1"],
                                _loaded["nAUT_3"],
                                _loaded["nAUT_5"],
                            )
                            per_seed_results.append(_loaded)
                            continue
                    except (json.JSONDecodeError, OSError):
                        pass

                logger.info("    [Seed %d] Forward-chaining %s (k=%s)…",
                            seed, model_family, horizon)
                set_global_seed(seed)
                t_seed = time.time()
                seed_artifacts_dir: Path | None = None
                if model_family == "transformer" and (
                    save_checkpoints or save_attention_artifacts
                ):
                    seed_artifacts_dir = horizon_dir / f"seed_{seed}"
                seed_result = _run_forward_chaining_seed(
                    model_family=model_family,
                    best_params=best_params,
                    n_features=n_features,
                    df=df,
                    year_col=year_col,
                    label_col=label_col,
                    feature_cols=feature_cols,
                    window_size=window_size,
                    batch_size=effective_batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    device=device,
                    seed=seed,
                    horizon=horizon,
                    num_workers=num_workers,
                    xgb_device=xgb_device,
                    timestamp_col=timestamp_col,
                    save_checkpoints=save_checkpoints,
                    save_attention_artifacts=save_attention_artifacts,
                    attention_samples_per_group=attention_samples_per_group,
                    seed_artifacts_dir=seed_artifacts_dir,
                )
                per_seed_results.append(seed_result)
                logger.info(
                    "    [Seed %d] nAUT_1=%.4f  nAUT_3=%.4f  nAUT_5=%.4f  (%.1fs)",
                    seed,
                    seed_result["nAUT_1"],
                    seed_result["nAUT_3"],
                    seed_result["nAUT_5"],
                    time.time() - t_seed,
                )
                # Save full seed result so any future failure can resume at seed level.
                save_results(seed_result, seed_json_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            aggregated = aggregate_metrics(
                per_seed_results, AXIS2_TEMPORAL_METRIC_KEYS + AXIS2_CLASS_METRIC_KEYS
            )
            elapsed_h = time.time() - t_h

            all_window_efficiency = [
                rec
                for sr in per_seed_results
                for rec in sr.get("window_efficiency", [])
            ]
            efficiency_summary = aggregate_efficiency_records(all_window_efficiency)

            h_result: dict[str, Any] = build_axis2_record(
                model_family=model_family,
                seeds=seeds,
                horizon=horizon,
                years=years,
                n_trials=n_trials,
                sample_frac=sample_frac,
                run_mode=run_mode,
                best_hpo_val_f1=search_result["best_value"],
                best_params=best_params,
                effective_batch_size=effective_batch_size,
                per_seed_results=per_seed_results,
                metrics=aggregated,
                efficiency_summary=efficiency_summary,
                save_checkpoints=save_checkpoints,
                save_attention_artifacts=save_attention_artifacts,
                attention_samples_per_group=(
                    attention_samples_per_group if save_attention_artifacts else None
                ),
                artifacts_dir=(
                    str(horizon_dir)
                    if (save_checkpoints or save_attention_artifacts)
                    and model_family == "transformer"
                    else None
                ),
                num_workers=num_workers,
                xgboost_device=xgb_device,
                hpo_device=hpo_device,
                elapsed_seconds=elapsed_h,
            )
            write_result(
                h_result, horizon_dir / "results.json", registry_dir=Path("data/results")
            )
            logger.info(
                "  [k=%s] Done %.1fs  nAUT_1_mean=%.4f  CI=[%.4f, %.4f]",
                horizon, elapsed_h,
                aggregated["nAUT_1"]["mean"],
                aggregated["nAUT_1"]["ci_lower"],
                aggregated["nAUT_1"]["ci_upper"],
            )
            horizon_results[h_key] = h_result

        except Exception as exc:  # noqa: BLE001
            logger.error("  [k=%s] FAILED: %s", horizon, exc, exc_info=True)
            error_payload = {
                "error": str(exc),
                "model_family": model_family,
                "horizon": str(horizon),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            try:
                save_results(error_payload, horizon_dir / "error.json")
            except Exception:  # noqa: BLE001
                pass
            horizon_results[h_key] = {"error": str(exc)}

            if fail_fast:
                raise RuntimeError(
                    f"--fail-fast: {comb} failed: {exc}"
                ) from exc

    return horizon_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Axis 2: MAWIFlow forward-chaining temporal evaluation."
    )
    parser.add_argument("--data-path", default="data/processed/mawiflow",
                        help="Path to MAWIFlow yearly parquet directory.")
    parser.add_argument("--output-dir", default="data/results/axis2")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    parser.add_argument(
        "--horizons", nargs="+", default=["1", "2", "3", "cumulative"],
        metavar="HORIZON",
        help="Forward-chaining horizons k (integers or 'cumulative'). Default: 1 2 3 cumulative.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS),
        metavar="SEED",
        help=(
            "Random seeds for repeated evaluation (R repetitions). "
            f"Default: {SEEDS}. "
            "Pass a single seed (e.g. --seeds 42) for fast pilot runs."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="List the execution matrix without training.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Subsample fraction (e.g. 0.05 for quick tests).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-checkpoints", action="store_true", default=False,
                        help="Save LightTransformer state_dict per seed/window "
                             "(transformer model only; off by default).")
    parser.add_argument("--save-attention-artifacts", action="store_true", default=False,
                        help="Generate attention heatmaps and attention_summary.json "
                             "per seed/window (transformer model only; off by default).")
    parser.add_argument("--attention-samples-per-group", type=int, default=16,
                        metavar="N",
                        help="Max correctly-classified samples per class collected for "
                             "attention averaging (default: 16).")
    _default_num_workers = 0 if sys.platform == "win32" else _compute_num_workers()
    parser.add_argument(
        "--num-workers", type=int, default=_default_num_workers,
        help=(
            "DataLoader worker processes for PyTorch models. "
            f"Default: {_default_num_workers} "
            f"({'0 on Windows to avoid WinError 1455 spawn failures' if sys.platform == 'win32' else 'auto'})."
        ),
    )
    parser.add_argument(
        "--xgboost-device", default="cpu", choices=["cpu", "cuda"],
        dest="xgboost_device",
        help="Device for XGBoost training (default: cpu). "
             "XGBoost runs on CPU by default even when CUDA is available to avoid "
             "device-mismatch warnings and to keep efficiency metrics comparable.",
    )
    # -- Resume / skip / fail-fast ------------------------------------------
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help=(
            "Reuse existing compatible results.json files. "
            "Incompatible results (different seeds/n_trials/sample_frac/run_mode) "
            "are re-run with a warning.  Combine with --fail-fast for strict mode."
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        dest="skip_existing",
        help=(
            "Like --resume but raises an error if an existing result is incompatible "
            "instead of re-running it.  Useful to guard against accidental overwrites."
        ),
    )
    parser.add_argument(
        "--fail-fast", action="store_true", default=False,
        dest="fail_fast",
        help=(
            "Abort the entire run on the first horizon failure instead of "
            "continuing with remaining horizons.  Prevents long partial runs."
        ),
    )
    args = parser.parse_args()

    horizons: list[int | str] = _parse_horizons(args.horizons)
    is_full = (
        sorted(args.models) == sorted(ALL_MODELS)
        and set(horizons) == set(ALL_HORIZONS)
    )
    is_full_seed_set = args.seeds == SEEDS
    run_mode = "production" if is_full_seed_set else "pilot"

    config: dict[str, Any] = {
        "dataset": {**_MAWIFLOW_CFG["dataset"], "path": args.data_path},
        "data": _MAWIFLOW_CFG["data"],
        "training": {
            **_MAWIFLOW_CFG["training"],
            **({"device": args.device} if args.device else {}),
        },
    }

    device = resolve_device(config)
    # cuDNN determinism is owned entirely by set_global_seed() (called per seed /
    # HPO inside run_model). Do not toggle torch.backends.cudnn.* here — a
    # benchmark=True set at this point was overridden before any training and
    # only obscured the deterministic protocol the project requires.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("axis2")

    n_combinations = len(args.models) * len(horizons)
    logger.info(
        "=== Axis 2: MAWIFlow  models=%s  horizons=%s  seeds=%s  combinations=%d  "
        "is_full=%s  run_mode=%s  device=%s  num_workers=%d  xgboost_device=%s "
        "resume=%s  skip_existing=%s  fail_fast=%s ===",
        args.models, horizons, args.seeds, n_combinations,
        is_full, run_mode, device, args.num_workers, args.xgboost_device,
        args.resume, args.skip_existing, args.fail_fast,
    )

    if args.dry_run:
        logger.info("=== DRY RUN — no training will be performed ===")
        logger.info("Execution matrix (%d combinations):", n_combinations)
        for model_family in args.models:
            for horizon in horizons:
                h_key = _horizon_dir_name(horizon)
                logger.info("  %s × k=%s → %s/%s/%s/results.json",
                            model_family, horizon, args.output_dir, model_family, h_key)
        logger.info("is_full_temporal_matrix: %s", is_full)
        logger.info("is_full_seed_set: %s", is_full_seed_set)
        logger.info("run_mode: %s", run_mode)
        return

    all_model_results: dict[str, dict[str, dict]] = {}
    combinations_executed: list[str] = []
    combinations_failed: list[str] = []
    combinations_skipped_existing: list[str] = []

    for model_family in args.models:
        try:
            model_results = run_model(
                model_family=model_family,
                config=config,
                sample_frac=args.sample_frac,
                n_trials=args.n_trials,
                max_epochs=args.max_epochs,
                horizons=horizons,
                device=device,
                output_dir=output_dir,
                logger=logger,
                seeds=args.seeds,
                save_checkpoints=args.save_checkpoints,
                save_attention_artifacts=args.save_attention_artifacts,
                attention_samples_per_group=args.attention_samples_per_group,
                num_workers=args.num_workers,
                xgb_device=args.xgboost_device,
                resume=args.resume,
                skip_existing=args.skip_existing,
                fail_fast=args.fail_fast,
                run_mode=run_mode,
            )
            all_model_results[model_family] = model_results
            for h_key, h_result in model_results.items():
                comb = f"{model_family}/{h_key}"
                if h_result.get("skipped_existing"):
                    combinations_skipped_existing.append(comb)
                elif "error" in h_result:
                    combinations_failed.append(comb)
                else:
                    combinations_executed.append(comb)
        except Exception as exc:
            logger.error("Model %s FAILED: %s", model_family, exc, exc_info=True)
            all_model_results[model_family] = {"error": str(exc)}
            for h in horizons:
                combinations_failed.append(f"{model_family}/{_horizon_dir_name(h)}")

    # -- Summary ------------------------------------------------------------
    summary_results: dict[str, Any] = {}
    for m, m_results in all_model_results.items():
        if "error" in m_results:
            summary_results[m] = {"error": m_results["error"]}
        else:
            summary_results[m] = {
                h_key: {
                    "horizon":            h_result.get("horizon"),
                    "metrics":            h_result.get("metrics", {}),
                    "best_params":        h_result.get("best_params", {}),
                    "efficiency_summary": h_result.get("efficiency_summary", {}),
                    "elapsed_seconds":    h_result.get("elapsed_seconds"),
                    "skipped_existing":   h_result.get("skipped_existing", False),
                    "hpo_device":         h_result.get("hpo_device"),
                    "xgboost_device":     h_result.get("xgboost_device"),
                }
                for h_key, h_result in m_results.items()
                if "error" not in h_result
            }
            failed_horizons = {
                h_key: h_result["error"]
                for h_key, h_result in m_results.items()
                if "error" in h_result
            }
            if failed_horizons:
                summary_results[m]["_failed_horizons"] = failed_horizons

    run_status = "completed" if not combinations_failed else "partial_failed"

    summary: dict[str, Any] = {
        "axis":                    2,
        "dataset":                 "MAWIFlow",
        "models":                  args.models,
        "horizons":                [str(h) for h in horizons],
        "seeds":                   args.seeds,
        "n_trials":                args.n_trials,
        "sample_frac":             args.sample_frac,
        "run_mode":                run_mode,
        "is_full_temporal_matrix": is_full,
        "is_full_seed_set":        is_full_seed_set,
        "combinations_executed":         combinations_executed,
        "combinations_skipped_existing": combinations_skipped_existing,
        "combinations_failed":           combinations_failed,
        "run_status":              run_status,
        # Execution metadata
        "num_workers":             args.num_workers,
        "xgboost_device":          args.xgboost_device,
        # Interpretability artifact configuration
        "save_checkpoints":          args.save_checkpoints,
        "save_attention_artifacts":  args.save_attention_artifacts,
        "attention_samples_per_group": args.attention_samples_per_group if args.save_attention_artifacts else None,
        "timestamp":               datetime.datetime.now().isoformat(timespec="seconds"),
        "results":                 summary_results,
    }
    save_results(summary, output_dir / "summary.json")
    logger.info("Summary → %s", output_dir / "summary.json")

    logger.info("\n=== Axis 2 Results ===")
    for m, m_results in all_model_results.items():
        if "error" in m_results:
            logger.info("  %-15s  FAILED: %s", m, m_results["error"])
            continue
        for h_key, h_result in m_results.items():
            if h_result.get("skipped_existing"):
                logger.info("  %-15s  %-14s  SKIPPED (existing)", m, h_key)
            elif "error" in h_result:
                logger.info("  %-15s  %-14s  FAILED: %s", m, h_key, h_result["error"])
            elif "metrics" in h_result:
                n1 = h_result["metrics"].get("nAUT_1", {})
                n3 = h_result["metrics"].get("nAUT_3", {})
                n5 = h_result["metrics"].get("nAUT_5", {})
                logger.info(
                    "  %-15s  %-14s  nAUT_1: %.4f±%.4f  nAUT_3: %.4f±%.4f  nAUT_5: %.4f±%.4f",
                    m, h_key,
                    n1.get("mean", float("nan")), n1.get("std", float("nan")),
                    n3.get("mean", float("nan")), n3.get("std", float("nan")),
                    n5.get("mean", float("nan")), n5.get("std", float("nan")),
                )


if __name__ == "__main__":
    main()
