#!/usr/bin/env python3
"""Axis 1: CICIoT2023 — 4 models × R=5 seeds with Optuna HPO + bootstrap CI.

Workflow:
  1. Load CICIoT2023 parquet.
  2. Static 60/20/20 split (fixed split seed for consistency across repetitions).
  3. For each model in [xgboost, mlp, cnn_bilstm, transformer]:
       a. Optuna HPO (n_trials, seed=SEEDS[0]).
       b. R=5 final training runs with seeds [42, 123, 456, 789, 1024].
       c. Bootstrap CI (B=1000) across seeds for each metric.
       d. Save per-seed + aggregated results to data/results/axis1/<model>/.
  4. Write summary to data/results/axis1/summary.json.

Usage:
    uv run python scripts/run_axis1.py
    uv run python scripts/run_axis1.py --sample-frac 0.05 --n-trials 10
    uv run python scripts/run_axis1.py --models xgboost mlp --device cuda
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_experiment import (
    _compute_num_workers,
    build_sequence_inputs,
    build_tabular_inputs,
    get_feature_cols,
    load_dataset,
    resolve_device,
)

from lift_nids.data.splits import static_split
from lift_nids.evaluation.metrics import compute_classification_metrics
from lift_nids.experiments.aggregation import aggregate_metrics
from lift_nids.experiments.constants import ALL_MODELS, AXIS1_METRIC_KEYS, SEEDS
from lift_nids.experiments.results import build_axis1_record, write_result
from lift_nids.training.family import make_model_family
from lift_nids.training.optuna_search import run_optuna_search
from lift_nids.utils.io import save_results
from lift_nids.utils.logging import setup_logger
from lift_nids.utils.reproducibility import set_global_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_SEED = 42  # Fixed so all 5 seeds share the same train/val/test split

_CICIOT_CFG: dict[str, Any] = {
    "dataset": {
        "name": "CICIoT2023",
        "path": "data/raw/CICIoT2023",
        "file_format": "parquet",
        "label_column": "label",
    },
    "data": {
        "split": {
            "train_size": 0.60,
            "val_size": 0.20,
            "test_size": 0.20,
            "stratify": True,
        },
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
# Helpers
# ---------------------------------------------------------------------------

def _train_and_evaluate(
    model_family: str,
    best_params: dict,
    n_features: int,
    train_loader,
    val_loader,
    test_loader,
    X_tr: np.ndarray | None,
    y_tr: np.ndarray | None,
    X_vl: np.ndarray | None,
    y_vl: np.ndarray | None,
    X_te: np.ndarray | None,
    y_te: np.ndarray | None,
    max_epochs: int,
    patience: int,
    device: str,
    seed: int,
    desc: str | None = None,
) -> dict[str, float]:
    """Rebuild model from best_params, train, and evaluate on test set."""
    family = make_model_family(
        model_family,
        best_params,
        n_features,
        device=device,
        seed=seed,
        patience=patience,
        max_epochs=max_epochs,
        use_class_weights=True,
        desc=desc,
    )
    if model_family == "xgboost":
        family.fit((X_tr, y_tr), (X_vl, y_vl))
        pred = family.predict((X_te, y_te))
    else:
        family.fit(train_loader, val_loader)
        pred = family.predict(test_loader)

    return compute_classification_metrics(pred.y_true, pred.y_pred, y_proba=pred.y_proba)


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_family: str,
    config: dict,
    sample_frac: float | None,
    n_trials: int,
    max_epochs: int,
    device: str,
    num_workers: int,
    output_dir: Path,
    logger,
) -> dict:
    """HPO + R=5 seeds + bootstrap CI for one model on CICIoT2023."""
    t0 = time.time()
    label_col = config["dataset"]["label_column"]
    patience = config["training"]["early_stopping"]["patience"]
    batch_size = config["training"]["batch_size"]
    window_size = config["data"]["temporal_sequence"]["window_size"]
    is_sequential = model_family in ("cnn_bilstm", "transformer")

    logger.info("=== Model: %s ===", model_family)
    logger.info("DataLoader workers: %d", num_workers)

    # -- Load + static split (fixed SPLIT_SEED across all 5 repetitions) ----
    logger.info("Loading CICIoT2023 (sample_frac=%s)…", sample_frac)
    df = load_dataset(config, sample_frac=sample_frac, seed=SPLIT_SEED)
    logger.info("Loaded %d rows  dist=%s", len(df),
                df[label_col].value_counts().to_dict())

    split_cfg = config["data"]["split"]
    train_df, val_df, test_df = static_split(
        df,
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        stratify_col=label_col if split_cfg["stratify"] else None,
        seed=SPLIT_SEED,
    )
    del df
    logger.info("Split: train=%d  val=%d  test=%d",
                len(train_df), len(val_df), len(test_df))

    feature_cols = get_feature_cols(train_df, label_col)

    # -- Build DataLoaders once (reused across all 5 seeds) -----------------
    if is_sequential:
        train_loader, val_loader, test_loader, n_features, _ = build_sequence_inputs(
            train_df, val_df, test_df, feature_cols, label_col,
            window_size=window_size, batch_size=batch_size,
            device=device, num_workers=num_workers,
        )
        X_tr = X_vl = X_te = y_tr = y_vl = y_te = None
        hpo_train_input, hpo_val_input = train_loader, val_loader
    else:
        _tab = build_tabular_inputs(
            train_df, val_df, test_df, feature_cols, label_col,
            batch_size=batch_size, device=device, num_workers=num_workers,
        )
        (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = _tab[0], _tab[1], _tab[2]
        train_loader, val_loader, test_loader = _tab[3], _tab[4], _tab[5]
        n_features = X_tr.shape[1]
        hpo_train_input = (X_tr, y_tr) if model_family == "xgboost" else train_loader
        hpo_val_input = (X_vl, y_vl) if model_family == "xgboost" else val_loader

    del train_df, val_df, test_df

    # -- Optuna HPO (once, with SEEDS[0]) -----------------------------------
    model_dir = output_dir / model_family
    model_dir.mkdir(parents=True, exist_ok=True)
    optuna_storage = f"sqlite:///{model_dir}/optuna.db"

    logger.info("[HPO] %d trials for %s…", n_trials, model_family)
    set_global_seed(SEEDS[0])
    search_result = run_optuna_search(
        model_type=model_family,
        train_data=hpo_train_input,
        val_data=hpo_val_input,
        n_features=n_features,
        n_classes=2,
        n_trials=n_trials,
        max_epochs=max_epochs,
        patience=patience,
        study_name=f"axis1_{model_family}",
        storage=optuna_storage,
        seed=SEEDS[0],
        device=device,
        use_class_weights=True,
    )
    best_params = search_result["best_params"]
    logger.info("[HPO] Best val F1=%.4f  params=%s",
                search_result["best_value"], best_params)

    # -- R=5 repetitions ----------------------------------------------------
    per_seed_results: list[dict] = []
    seed_pbar = tqdm(
        SEEDS, desc=f"Seeds [{model_family}]", unit="seed", dynamic_ncols=True, leave=True
    )
    for seed in seed_pbar:
        seed_pbar.set_postfix({"seed": seed}, refresh=True)
        logger.info("[Seed %d] Training %s…", seed, model_family)
        set_global_seed(seed)
        metrics = _train_and_evaluate(
            model_family=model_family,
            best_params=best_params,
            n_features=n_features,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            X_tr=X_tr, y_tr=y_tr,
            X_vl=X_vl, y_vl=y_vl,
            X_te=X_te, y_te=y_te,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            seed=seed,
            desc=f"{model_family} seed={seed}",
        )
        per_seed_results.append({"seed": seed, **metrics})
        logger.info(
            "[Seed %d] F1_1=%.4f  F1_macro=%.4f  AUC-ROC=%.4f",
            seed, metrics["f1_1"], metrics["f1_macro"], metrics["auc_roc"],
        )
        save_results(
            {"seed": seed, "model": model_family, **metrics},
            model_dir / f"seed_{seed}.json",
        )

    # -- Bootstrap CI -------------------------------------------------------
    aggregated = aggregate_metrics(per_seed_results, AXIS1_METRIC_KEYS)
    elapsed = time.time() - t0

    result: dict[str, Any] = build_axis1_record(
        model_family=model_family,
        seeds=SEEDS,
        n_trials=n_trials,
        num_workers=num_workers,
        effective_batch_size=batch_size,
        best_hpo_val_f1=search_result["best_value"],
        best_params=best_params,
        per_seed_metrics=per_seed_results,
        metrics=aggregated,
        elapsed_seconds=elapsed,
    )
    write_result(result, model_dir / "results.json", registry_dir=Path("data/results"))
    logger.info(
        "[Done] %s  %.1fs  F1+_mean=%.4f  CI=[%.4f, %.4f]",
        model_family, elapsed,
        aggregated["f1_1"]["mean"],
        aggregated["f1_1"]["ci_lower"],
        aggregated["f1_1"]["ci_upper"],
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Axis 1: CICIoT2023 — 4 models × R=5 seeds."
    )
    parser.add_argument("--data-path", default="data/raw/CICIoT2023",
                        help="Path to CICIoT2023 parquet directory.")
    parser.add_argument("--output-dir", default="data/results/axis1")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Subsample fraction (e.g. 0.05 for quick tests).")
    parser.add_argument("--device", default=None)
    _default_num_workers = 0 if sys.platform == "win32" else _compute_num_workers()
    parser.add_argument(
        "--num-workers", type=int, default=_default_num_workers,
        help=(
            "DataLoader worker processes for PyTorch models. "
            f"Default: {_default_num_workers} "
            "(0 on Windows to avoid spawn/pickle failures with large arrays)."
        ),
    )
    args = parser.parse_args()

    config: dict[str, Any] = {
        "dataset": {**_CICIOT_CFG["dataset"], "path": args.data_path},
        "data": _CICIOT_CFG["data"],
        "training": {
            **_CICIOT_CFG["training"],
            **({"device": args.device} if args.device else {}),
        },
    }

    device = resolve_device(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("axis1")
    logger.info("=== Axis 1: CICIoT2023  models=%s  device=%s  num_workers=%d ===",
                args.models, device, args.num_workers)

    all_results: dict[str, dict] = {}
    for model_family in args.models:
        try:
            all_results[model_family] = run_model(
                model_family=model_family,
                config=config,
                sample_frac=args.sample_frac,
                n_trials=args.n_trials,
                max_epochs=args.max_epochs,
                device=device,
                num_workers=args.num_workers,
                output_dir=output_dir,
                logger=logger,
            )
        except Exception as exc:
            logger.error("Model %s FAILED: %s", model_family, exc, exc_info=True)
            all_results[model_family] = {"error": str(exc)}

    summary: dict[str, Any] = {
        "axis":      1,
        "dataset":   "CICIoT2023",
        "models":    args.models,
        "seeds":     SEEDS,
        "n_trials":  args.n_trials,
        "num_workers": args.num_workers,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "results": {
            m: {
                "metrics":         r.get("metrics", {}),
                "best_params":     r.get("best_params", {}),
                "elapsed_seconds": r.get("elapsed_seconds"),
            }
            for m, r in all_results.items()
        },
    }
    save_results(summary, output_dir / "summary.json")
    logger.info("Summary → %s", output_dir / "summary.json")

    logger.info("\n=== Axis 1 Results ===")
    for m, r in all_results.items():
        if "error" in r:
            logger.info("  %-15s  FAILED: %s", m, r["error"])
        elif "metrics" in r:
            f1 = r["metrics"].get("f1_1", {})
            auc = r["metrics"].get("auc_roc", {})
            logger.info(
                "  %-15s  F1+: %.4f±%.4f [%.4f,%.4f]  AUC: %.4f±%.4f",
                m,
                f1.get("mean", float("nan")), f1.get("std", float("nan")),
                f1.get("ci_lower", float("nan")), f1.get("ci_upper", float("nan")),
                auc.get("mean", float("nan")), auc.get("std", float("nan")),
            )


if __name__ == "__main__":
    main()
