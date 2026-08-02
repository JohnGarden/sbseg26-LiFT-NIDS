"""Computational efficiency metrics for model comparison.

Provides helpers for counting parameters, timing, GPU memory tracking, and
building the standardized ``efficiency`` record that is persisted alongside
classification metrics in every results.json produced by the LiFT-NIDS runners.

Fields in an efficiency record
-------------------------------
n_params                  : int   — trainable parameters (PyTorch) or
                                    n_estimators × max_depth proxy (XGBoost)
train_time_s              : float — wall-clock fit time in seconds
infer_time_s              : float — wall-clock inference time for n_infer_samples
infer_time_s_per_sample   : float — infer_time_s / n_infer_samples
peak_gpu_mem_bytes        : int   — peak GPU memory allocated; 0 for CPU runs
device                    : str   — device string (e.g. "cpu", "cuda:0")
n_train_samples           : int   — number of training rows passed to fit
n_infer_samples           : int   — number of rows used for the inference measurement
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_trainable_params(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model.

    Args:
        model: Any ``nn.Module`` instance.

    Returns:
        Sum of ``numel()`` for all parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------

def _to_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(str(device))


def sync_cuda_if_needed(device: str | torch.device) -> None:
    """Synchronize CUDA if ``device`` is a CUDA device and CUDA is available.

    Must be called before and after timed inference to obtain accurate wall-clock
    measurements; no-op on CPU or when CUDA is unavailable.
    """
    dev = _to_device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def measure_pytorch_inference(
    predict_fn: Callable,
    model: torch.nn.Module,
    loader: Any,
    device: str | torch.device,
) -> tuple:
    """Run ``predict_fn`` with CUDA-synchronized wall-clock timing.

    Calls :func:`sync_cuda_if_needed` before starting and after stopping the
    clock so that any queued CUDA kernels complete before the measurement is
    read — a no-op on CPU.

    Args:
        predict_fn: Callable with signature ``(model, loader, device) ->
            (y_true, y_pred, y_proba)``.
        model:      PyTorch model in eval mode.
        loader:     DataLoader to iterate over.
        device:     Device string or ``torch.device``.

    Returns:
        ``(y_true, y_pred, y_proba, elapsed_s)`` where ``elapsed_s`` is the
        synchronized wall-clock inference time in seconds.
    """
    sync_cuda_if_needed(device)
    t0 = time.perf_counter()
    y_true, y_pred, y_proba = predict_fn(model, loader, device)
    sync_cuda_if_needed(device)
    elapsed = time.perf_counter() - t0
    return y_true, y_pred, y_proba, elapsed


def get_peak_gpu_memory_bytes(device: str | torch.device) -> int:
    """Return peak GPU memory allocated (bytes) since the last reset, or 0 for CPU."""
    dev = _to_device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(dev)
    return 0


def reset_peak_gpu_memory(device: str | torch.device) -> None:
    """Reset peak GPU memory statistics for ``device``; no-op on CPU."""
    dev = _to_device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def build_efficiency_record(
    n_params: int,
    train_time_s: float,
    infer_time_s: float,
    n_train_samples: int,
    n_infer_samples: int,
    peak_gpu_mem_bytes: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Build a standardized efficiency record for one training/inference run.

    Args:
        n_params:           Trainable parameter count.
        train_time_s:       Wall-clock fit duration in seconds.
        infer_time_s:       Wall-clock inference duration for ``n_infer_samples``.
        n_train_samples:    Number of rows used for fitting.
        n_infer_samples:    Number of rows used for the inference measurement.
        peak_gpu_mem_bytes: Peak GPU memory; use 0 for CPU-only models.
        device:             Device string or ``torch.device``.

    Returns:
        Dict with all eight standardized efficiency fields.
    """
    per_sample: float | None = (
        infer_time_s / n_infer_samples if n_infer_samples > 0 else None
    )
    return {
        "n_params":                n_params,
        "train_time_s":            round(float(train_time_s), 4),
        "infer_time_s":            round(float(infer_time_s), 6),
        "infer_time_s_per_sample": round(per_sample, 9) if per_sample is not None else None,
        "peak_gpu_mem_bytes":      int(peak_gpu_mem_bytes),
        "device":                  str(device),
        "n_train_samples":         int(n_train_samples),
        "n_infer_samples":         int(n_infer_samples),
    }


_REQUIRED_FIELDS = frozenset({
    "n_params", "train_time_s", "infer_time_s", "infer_time_s_per_sample",
    "peak_gpu_mem_bytes", "device", "n_train_samples", "n_infer_samples",
})


def aggregate_efficiency_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate a list of efficiency records into a summary.

    Returns mean and median of ``train_time_s`` and ``infer_time_s_per_sample``,
    ``n_params`` (constant across records — taken from first), and
    ``max_peak_gpu_mem_bytes``.

    Args:
        records: List of dicts produced by :func:`build_efficiency_record`.

    Returns:
        Summary dict, or empty dict when ``records`` is empty.
    """
    if not records:
        return {}

    def _vals(key: str) -> list[float]:
        return [r[key] for r in records if r.get(key) is not None]

    train_times      = _vals("train_time_s")
    infer_per_sample = _vals("infer_time_s_per_sample")
    peak_mems        = _vals("peak_gpu_mem_bytes")

    return {
        "n_params":                            records[0].get("n_params"),
        "device":                              records[0].get("device"),
        "mean_train_time_s":                   float(np.mean(train_times))      if train_times      else None,
        "median_train_time_s":                 float(np.median(train_times))    if train_times      else None,
        "mean_infer_time_s_per_sample":        float(np.mean(infer_per_sample)) if infer_per_sample else None,
        "median_infer_time_s_per_sample":      float(np.median(infer_per_sample)) if infer_per_sample else None,
        "max_peak_gpu_mem_bytes":              int(max(peak_mems))              if peak_mems        else 0,
        "n_records":                           len(records),
    }
