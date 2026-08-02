"""Integration tests for build_sequence_inputs in scripts/run_experiment.py.

Validates that the function uses SlidingWindowBuilder (zero-padding + correct
strides) rather than windows_from_dataframe (no padding, stride=1 for all splits).

The key behavioral differences that distinguish the implementations:

  Implementation         | train windows (N=20, W=8) | val windows (N=15)
  ---------------------- | ------------------------- | ------------------
  windows_from_dataframe | 13  (no pad, stride=1)   | 8  (no pad, stride=1)
  SlidingWindowBuilder   |  5  (pad, stride=W//2=4) | 15 (pad, stride=1)

These tests fail against the old implementation and pass after the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make run_experiment importable (it adds src/ to sys.path on import).
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest

from run_experiment import build_sequence_inputs  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

W = 8          # smallest valid window_size in SlidingWindowBuilder
N_TRAIN = 20
N_VAL = 15
N_TEST = 15
N_FEAT = 3     # number of numerical features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.random(n).astype(np.float32) for i in range(N_FEAT)}
    data["label"] = rng.integers(0, 2, size=n).astype(np.int64)
    # Guarantee both classes present so the preprocessor / metrics don't fail.
    data["label"][0] = 0
    data["label"][1] = 1
    return pd.DataFrame(data)


def _call(train_df, val_df, test_df, window_size=W):
    feat_cols = [c for c in train_df.columns if c != "label"]
    return build_sequence_inputs(
        train_df, val_df, test_df,
        feature_cols=(feat_cols, []),
        label_col="label",
        window_size=window_size,
        batch_size=64,
    )


def _count(loader) -> int:
    """Total number of samples across all batches in a DataLoader."""
    return sum(y.shape[0] for _, y in loader)


def _collect_labels(loader) -> np.ndarray:
    parts = [y.numpy() for _, y in loader]
    return np.concatenate(parts)


def _first_window(loader) -> np.ndarray:
    """Return the first window tensor (W, F) from the first batch."""
    X, _ = next(iter(loader))
    return X[0].numpy()          # (W, F)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(name="splits")
def _splits():
    return _make_df(N_TRAIN, seed=0), _make_df(N_VAL, seed=1), _make_df(N_TEST, seed=2)


# ---------------------------------------------------------------------------
# Return-contract tests
# ---------------------------------------------------------------------------

class TestReturnContract:
    def test_returns_five_elements(self, splits):
        result = _call(*splits)
        assert len(result) == 5

    def test_n_features_is_positive_int(self, splits):
        *_, n_features, _ = _call(*splits)
        assert isinstance(n_features, int) and n_features > 0

    def test_preprocessor_is_not_none(self, splits):
        *_, preprocessor = _call(*splits)
        assert preprocessor is not None


# ---------------------------------------------------------------------------
# Window-count tests (distinguish SlidingWindowBuilder from windows_from_dataframe)
# ---------------------------------------------------------------------------

class TestWindowCounts:
    def test_train_uses_stride_half_w(self, splits):
        # SlidingWindowBuilder.build_train: stride = W//2
        # Expected: len(range(0, N_TRAIN, W//2)) windows
        # Old implementation (windows_from_dataframe, stride=1):
        #   N_TRAIN - W + 1 = 13 windows  ← would fail this assertion
        train_loader, _, _, _, _ = _call(*splits)
        expected = len(range(0, N_TRAIN, W // 2))  # 5
        assert _count(train_loader) == expected

    def test_val_one_window_per_flow(self, splits):
        # SlidingWindowBuilder.build_test: stride=1, pad to include every flow
        # Expected: N_VAL windows
        # Old implementation: N_VAL - W + 1 = 8 windows  ← would fail
        _, val_loader, _, _, _ = _call(*splits)
        assert _count(val_loader) == N_VAL

    def test_test_one_window_per_flow(self, splits):
        _, _, test_loader, _, _ = _call(*splits)
        assert _count(test_loader) == N_TEST


# ---------------------------------------------------------------------------
# Padding tests
# ---------------------------------------------------------------------------

class TestPadding:
    def test_val_first_window_has_w_minus_1_zero_rows(self, splits):
        # With SlidingWindowBuilder, the very first window is all-zero except
        # the last row (the first data flow).  Without padding, the first window
        # would start at flow index W-1 and contain no zeros.
        _, val_loader, _, _, _ = _call(*splits)
        first = _first_window(val_loader)       # (W, F)
        np.testing.assert_array_equal(
            first[: W - 1],
            np.zeros((W - 1, first.shape[-1]), dtype=np.float32),
        )

    def test_test_first_window_has_w_minus_1_zero_rows(self, splits):
        _, _, test_loader, _, _ = _call(*splits)
        first = _first_window(test_loader)
        np.testing.assert_array_equal(
            first[: W - 1],
            np.zeros((W - 1, first.shape[-1]), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Label-alignment tests
# ---------------------------------------------------------------------------

class TestLabelAlignment:
    def test_val_labels_match_original_order(self, splits):
        # stride=1 → window i has label[i]
        train_df, val_df, test_df = splits
        _, val_loader, _, _, _ = _call(train_df, val_df, test_df)
        actual = _collect_labels(val_loader)
        expected = val_df["label"].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(actual, expected)

    def test_test_labels_match_original_order(self, splits):
        train_df, val_df, test_df = splits
        _, _, test_loader, _, _ = _call(train_df, val_df, test_df)
        actual = _collect_labels(test_loader)
        expected = test_df["label"].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Small-split robustness (fewer rows than W)
# ---------------------------------------------------------------------------

class TestSmallSplits:
    def test_val_smaller_than_window_size(self, splits):
        train_df, val_df, test_df = splits
        small_val = val_df.iloc[:2].reset_index(drop=True)
        _, val_loader, _, _, _ = _call(train_df, small_val, test_df)
        assert _count(val_loader) == 2

    def test_test_single_row(self, splits):
        train_df, val_df, test_df = splits
        single = test_df.iloc[:1].reset_index(drop=True)
        _, _, test_loader, _, _ = _call(train_df, val_df, single)
        assert _count(test_loader) == 1

    def test_train_single_row(self, splits):
        train_df, val_df, test_df = splits
        single_train = train_df.iloc[:1].reset_index(drop=True)
        # Guarantee both classes present in single row (label=1 already set above)
        single_train = single_train.copy()
        single_train["label"] = 1
        # Even one train row should produce exactly one window
        train_loader, _, _, _, _ = _call(single_train, val_df, test_df)
        assert _count(train_loader) == 1
