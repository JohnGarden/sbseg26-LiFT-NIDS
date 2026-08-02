"""Batched inference for PyTorch classifiers.

Single source of truth for turning a trained model + DataLoader into aligned
``(y_true, y_pred, y_proba)`` arrays. Used by the ModelFamily adapters and,
via re-export from the experiment scripts, by the forward-chaining runner.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_pytorch(
    model: torch.nn.Module,
    loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred, y_proba)`` where ``y_proba`` is P(class 1).

    Args:
        model: Trained classifier producing ``(batch, n_classes)`` logits.
        loader: DataLoader yielding ``(X, y)`` batches.
        device: Device string the model lives on.

    Returns:
        Three 1-D arrays aligned by sample: ground-truth labels, argmax
        predictions, and positive-class probabilities.
    """
    model.eval()
    all_true, all_pred, all_proba = [], [], []
    for X, y in loader:
        logits = model(X.to(device))
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        all_true.append(y.numpy())
        all_pred.append(pred)
        all_proba.append(proba)
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_proba)
