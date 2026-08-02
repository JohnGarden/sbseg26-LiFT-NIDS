"""Unified training loop for all PyTorch models in LiFT-NIDS."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lift_nids.training.early_stopping import EarlyStopping


class Trainer:
    """Unified trainer for MLP, CNN-BiLSTM, and LightTransformer.

    Handles the training loop, validation, early stopping, and checkpointing.
    XGBoostWrapper is not managed here — use XGBoostWrapper.fit() directly.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str | torch.device = "cpu",
        early_stopping: EarlyStopping | None = None,
        n_classes: int = 2,
        monitor: str = "f1_1",
        log_train_metrics: bool = True,
        use_amp: bool = True,
    ) -> None:
        """
        Args:
            monitor: Validation metric used by early stopping and checkpointing.
                     One of 'f1_macro', 'f1_1', 'loss'. Default 'f1_1' (F1+).
            log_train_metrics: When False, skip per-epoch train sklearn metrics
                (f1_macro, f1_1, precision, recall).  Early stopping and val
                metrics are unaffected.  Set to False in production runs to
                avoid D2H copies + sklearn compute on large training sets.
            use_amp: Enable automatic mixed precision (FP16) on CUDA devices.
                     Ignored on CPU. Provides ~1.5-2x speedup on tensor-core GPUs.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.early_stopping = early_stopping
        self.n_classes = n_classes
        self.monitor = monitor
        self._log_train_metrics = log_train_metrics
        # Enable async host-to-device transfers when pin_memory=True in DataLoader
        self._non_blocking = self.device.type == "cuda"
        # AMP: only active on CUDA
        self._use_amp = use_amp and self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

    # ------------------------------------------------------------------
    # Core epoch methods
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """One full pass over train_loader. Returns loss, f1_macro, f1_1, precision, recall."""
        self.model.train()
        total_loss = 0.0
        n_total = 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for X, y in loader:
            X = X.to(self.device, non_blocking=self._non_blocking)
            y = y.to(self.device, non_blocking=self._non_blocking)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                logits = self.model(X)
                loss = self.criterion(logits, y)
            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()

            batch_n = len(y)
            total_loss += loss.item() * batch_n
            n_total += batch_n
            if self._log_train_metrics:
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        if self._log_train_metrics:
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            return {
                "loss": total_loss / n_total,
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
                "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            }
        return {
            "loss": total_loss / n_total,
            "f1_macro": 0.0, "f1_1": 0.0, "precision": 0.0, "recall": 0.0,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate model on loader. Returns loss, f1_macro, f1_1, precision, recall."""
        self.model.eval()
        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for X, y in loader:
            X = X.to(self.device, non_blocking=self._non_blocking)
            y = y.to(self.device, non_blocking=self._non_blocking)
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                logits = self.model(X)
                loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        return {
            "loss": total_loss / len(y_true),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 30,
        checkpoint_dir: Path | None = None,
        epoch_callback: Callable[[int, dict[str, float]], bool] | None = None,
        desc: str | None = None,
    ) -> dict[str, Any]:
        """Train until max_epochs or early stopping triggers.

        Args:
            epoch_callback: Called after each validation pass as
                ``callback(epoch, val_metrics) -> should_prune``.
                Return True to stop training early (used by Optuna pruning).

        Returns:
            History dict with per-epoch lists for train/val loss, F1, F1+,
            precision, and recall, plus scalar 'train_time_s' and
            'peak_gpu_mem_bytes'.
        """
        history: dict[str, Any] = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "train_f1_1": [], "val_f1_1": [],
            "train_precision": [], "val_precision": [],
            "train_recall": [], "val_recall": [],
        }

        best_path = (checkpoint_dir / "best.pt") if checkpoint_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        t_start = time.perf_counter()

        epoch_iter: Any = range(1, max_epochs + 1)
        pbar: tqdm | None = None
        if desc is not None:
            pbar = tqdm(epoch_iter, desc=desc, unit="ep", dynamic_ncols=True, leave=True)
            epoch_iter = pbar

        for epoch in epoch_iter:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_f1"].append(train_metrics["f1_macro"])
            history["val_f1"].append(val_metrics["f1_macro"])
            history["train_f1_1"].append(train_metrics["f1_1"])
            history["val_f1_1"].append(val_metrics["f1_1"])
            history["train_precision"].append(train_metrics["precision"])
            history["val_precision"].append(val_metrics["precision"])
            history["train_recall"].append(train_metrics["recall"])
            history["val_recall"].append(val_metrics["recall"])

            if pbar is not None:
                pbar.set_postfix({
                    "val_f1+": f"{val_metrics['f1_1']:.4f}",
                    "val_loss": f"{val_metrics['loss']:.4f}",
                }, refresh=True)

            # Optuna pruning hook
            if epoch_callback is not None and epoch_callback(epoch, val_metrics):
                break

            monitor_val = val_metrics.get(self.monitor, val_metrics["f1_1"])

            if self.early_stopping is not None:
                stop = self.early_stopping.step(monitor_val, self.model)
                if best_path and self.early_stopping.improved:
                    self.save_checkpoint(best_path)
                if stop:
                    if self.early_stopping.restore_best:
                        self.early_stopping.restore(self.model)
                    break
            elif best_path:
                self.save_checkpoint(best_path)

        if pbar is not None:
            pbar.close()

        history["train_time_s"] = time.perf_counter() - t_start
        history["peak_gpu_mem_bytes"] = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )

        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])


# ------------------------------------------------------------------
# Helper factories
# ------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Factory for common optimizers."""
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name!r}. Choose adam | adamw | sgd.")


def compute_class_weights(
    labels: Any,
    n_classes: int = 2,
) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced datasets.

    Returns a float tensor of shape (n_classes,) suitable for
    nn.CrossEntropyLoss(weight=...).
    """
    labels_arr = np.asarray(labels).ravel()
    counts = np.bincount(labels_arr, minlength=n_classes).astype(float)
    # Avoid division by zero for classes absent from this split
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(labels_arr) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
