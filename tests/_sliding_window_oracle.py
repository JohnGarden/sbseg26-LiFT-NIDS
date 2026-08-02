"""Materialized sliding-window reference implementation — TEST ORACLE ONLY.

This is the eager, fully-materialized windowing that ``LazySequenceFlowDataset``
replaced in production. It is kept solely as an independent reference so the
lazy dataset can be checked for equivalence (``test_lazy_sequence_dataset.py``)
and so the windowing arithmetic itself has hand-computed coverage
(``test_sliding_window.py``). It is NOT importable from ``lift_nids`` and is not
collected as a test module (leading-underscore filename).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW_SIZE: int = 16

VALID_WINDOW_SIZES: frozenset[int] = frozenset({8, 16, 32})


def create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a sequence of flow records.

    Args:
        features: Float array of shape (N, n_features).
        labels: Integer array of shape (N,).
        window_size: Number of consecutive flows per window.
        stride: Step between window starts.

    Returns:
        windows: np.ndarray of shape (M, window_size, n_features)
        window_labels: np.ndarray of shape (M,) — label of last record in each window
    """
    n = len(features)
    if n < window_size:
        raise ValueError(
            f"Sequence length {n} is shorter than window_size {window_size}."
        )

    indices = range(0, n - window_size + 1, stride)
    windows = np.stack([features[i : i + window_size] for i in indices])
    window_labels = np.array([labels[i + window_size - 1] for i in indices])
    return windows, window_labels


def windows_from_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = 1,
    sort_col: str | None = None,
    group_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """High-level wrapper: optionally sort and/or group before windowing.

    Args:
        df: Input DataFrame.
        feature_cols: Columns to use as features.
        label_col: Column to use as label.
        window_size: Window size.
        stride: Step between windows.
        sort_col: If provided, sort df by this column before windowing.
        group_col: If provided, create windows within each group independently.

    Returns:
        windows: (M, window_size, n_features)
        labels: (M,)
    """
    if sort_col:
        df = df.sort_values(sort_col)

    if group_col is None:
        feats = df[feature_cols].to_numpy(dtype=np.float32)
        labs = df[label_col].to_numpy(dtype=np.int64)
        return create_windows(feats, labs, window_size, stride)

    all_windows, all_labels = [], []
    for _, group in df.groupby(group_col, sort=False):
        if sort_col:
            group = group.sort_values(sort_col)
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        labs = group[label_col].to_numpy(dtype=np.int64)
        if len(feats) < window_size:
            continue
        win, win_labs = create_windows(feats, labs, window_size, stride)
        all_windows.append(win)
        all_labels.append(win_labs)

    if not all_windows:
        n_feat = len(feature_cols)
        return np.empty((0, window_size, n_feat), dtype=np.float32), np.empty(0, dtype=np.int64)

    return np.concatenate(all_windows), np.concatenate(all_labels)


class SlidingWindowBuilder:
    """Builds sliding window sequences from a chronologically ordered DataFrame.

    Applies zero-padding on the left so every flow record (including the first
    W-1) gets its own window. Label for each window is the label of the last
    (most recent) flow — the Last Token head convention.

    Args:
        feature_cols: Ordered list of feature column names.
        label_col: Name of the label column.
        window_size: Number of flows per window. Must be in {8, 16, 32}.
    """

    def __init__(
        self,
        feature_cols: list[str],
        label_col: str,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> None:
        if window_size not in VALID_WINDOW_SIZES:
            raise ValueError(
                f"window_size must be one of {sorted(VALID_WINDOW_SIZES)}, got {window_size}"
            )
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size

    def _build(self, df: pd.DataFrame, stride: int) -> tuple[np.ndarray, np.ndarray]:
        W = self.window_size
        features = df[self.feature_cols].to_numpy(dtype=np.float32)
        labels = df[self.label_col].to_numpy(dtype=np.int64)
        n, n_feat = features.shape

        pad = np.zeros((W - 1, n_feat), dtype=np.float32)
        padded = np.concatenate([pad, features], axis=0)

        indices = range(0, n, stride)
        windows = np.stack([padded[i : i + W] for i in indices])
        window_labels = np.array([labels[i] for i in indices], dtype=np.int64)
        return windows, window_labels

    def build_train(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Build training sequences with stride = W // 2."""
        return self._build(df, stride=self.window_size // 2)

    def build_test(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Build test sequences with stride = 1."""
        return self._build(df, stride=1)
