"""Tests for LazySequenceFlowDataset.

Verifies equivalence with SlidingWindowBuilder and covers padding,
label semantics, shape, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from _sliding_window_oracle import SlidingWindowBuilder

from lift_nids.data.dataset import LazySequenceFlowDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 40, n_feat: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.random(n).astype(np.float32) for i in range(n_feat)}
    data["label"] = rng.integers(0, 2, n).astype(np.int64)
    return pd.DataFrame(data)


def _df_to_arrays(
    df: pd.DataFrame, feat_cols: list[str], label_col: str
) -> tuple[np.ndarray, np.ndarray]:
    return df[feat_cols].to_numpy(np.float32), df[label_col].to_numpy(np.int64)


# ---------------------------------------------------------------------------
# Equivalence with SlidingWindowBuilder
# ---------------------------------------------------------------------------

class TestEquivalenceWithSlidingWindowBuilder:
    """LazySequenceFlowDataset must produce identical windows and labels to
    SlidingWindowBuilder for both training (stride=W//2) and test (stride=1)."""

    _FEAT = ["f0", "f1", "f2"]
    _LABEL = "label"
    _W = 8  # must be in {8, 16, 32} for SlidingWindowBuilder

    def _ref_train(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return SlidingWindowBuilder(self._FEAT, self._LABEL, self._W).build_train(df)

    def _ref_test(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return SlidingWindowBuilder(self._FEAT, self._LABEL, self._W).build_test(df)

    def test_train_length_matches(self) -> None:
        df = _make_df(60)
        _, ref_labs = self._ref_train(df)
        features, labels = _df_to_arrays(df, self._FEAT, self._LABEL)
        lazy = LazySequenceFlowDataset(features, labels, self._W, stride=self._W // 2)
        assert len(lazy) == len(ref_labs)

    def test_test_length_matches(self) -> None:
        df = _make_df(60)
        _, ref_labs = self._ref_test(df)
        features, labels = _df_to_arrays(df, self._FEAT, self._LABEL)
        lazy = LazySequenceFlowDataset(features, labels, self._W, stride=1)
        assert len(lazy) == len(ref_labs)

    def test_train_windows_match(self) -> None:
        df = _make_df(60)
        ref_wins, _ = self._ref_train(df)
        features, labels = _df_to_arrays(df, self._FEAT, self._LABEL)
        lazy = LazySequenceFlowDataset(features, labels, self._W, stride=self._W // 2)
        for idx in range(len(lazy)):
            window, _ = lazy[idx]
            np.testing.assert_array_almost_equal(window.numpy(), ref_wins[idx], decimal=6)

    def test_test_windows_match(self) -> None:
        df = _make_df(60)
        ref_wins, _ = self._ref_test(df)
        features, labels = _df_to_arrays(df, self._FEAT, self._LABEL)
        lazy = LazySequenceFlowDataset(features, labels, self._W, stride=1)
        for idx in range(len(lazy)):
            window, _ = lazy[idx]
            np.testing.assert_array_almost_equal(window.numpy(), ref_wins[idx], decimal=6)

    def test_train_labels_match(self) -> None:
        df = _make_df(60)
        _, ref_labs = self._ref_train(df)
        features, labels = _df_to_arrays(df, self._FEAT, self._LABEL)
        lazy = LazySequenceFlowDataset(features, labels, self._W, stride=self._W // 2)
        lazy_labs = np.array([lazy[i][1].item() for i in range(len(lazy))], dtype=np.int64)
        np.testing.assert_array_equal(lazy_labs, ref_labs)

    def test_test_labels_match(self) -> None:
        df = _make_df(60)
        _, ref_labs = self._ref_test(df)
        features, labels = _df_to_arrays(df, self._FEAT, self._LABEL)
        lazy = LazySequenceFlowDataset(features, labels, self._W, stride=1)
        lazy_labs = np.array([lazy[i][1].item() for i in range(len(lazy))], dtype=np.int64)
        np.testing.assert_array_equal(lazy_labs, ref_labs)

    def test_window_size_16_equivalence(self) -> None:
        """Verify with the default W=16 used in production."""
        W = 16
        df = _make_df(80)
        ref_wins, ref_labs = SlidingWindowBuilder(
            self._FEAT, self._LABEL, W
        ).build_test(df)
        features, labels = _df_to_arrays(df, self._FEAT, self._LABEL)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        assert len(lazy) == len(ref_labs)
        for idx in range(len(lazy)):
            window, label = lazy[idx]
            np.testing.assert_array_almost_equal(window.numpy(), ref_wins[idx], decimal=6)
            assert label.item() == ref_labs[idx]


# ---------------------------------------------------------------------------
# Padding and label semantics
# ---------------------------------------------------------------------------

class TestPaddingAndLabelSemantics:
    def test_first_window_has_w_minus_1_zero_rows(self) -> None:
        n, W, F = 30, 8, 3
        rng = np.random.default_rng(0)
        features = rng.random((n, F)).astype(np.float32)
        labels = rng.integers(0, 2, n).astype(np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        window, _ = lazy[0]
        np.testing.assert_array_equal(
            window[: W - 1].numpy(), np.zeros((W - 1, F), dtype=np.float32)
        )

    def test_first_window_last_row_is_first_data_row(self) -> None:
        n, W, F = 30, 8, 3
        rng = np.random.default_rng(0)
        features = rng.random((n, F)).astype(np.float32)
        labels = rng.integers(0, 2, n).astype(np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        window, _ = lazy[0]
        np.testing.assert_array_almost_equal(window[W - 1].numpy(), features[0])

    def test_padding_decreases_by_one_per_window(self) -> None:
        n, W, F = 20, 8, 2
        rng = np.random.default_rng(1)
        features = rng.random((n, F)).astype(np.float32)
        labels = rng.integers(0, 2, n).astype(np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        for i in range(min(W, n)):
            window, _ = lazy[i]
            n_zeros = W - 1 - i
            if n_zeros > 0:
                np.testing.assert_array_equal(
                    window[:n_zeros].numpy(),
                    np.zeros((n_zeros, F), dtype=np.float32),
                )
            # Last row of window is features[i]
            np.testing.assert_array_almost_equal(window[W - 1].numpy(), features[i])

    def test_window_at_index_w_minus_1_has_no_padding(self) -> None:
        n, W, F = 30, 8, 3
        rng = np.random.default_rng(2)
        features = rng.random((n, F)).astype(np.float32)
        labels = rng.integers(0, 2, n).astype(np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        window, _ = lazy[W - 1]
        np.testing.assert_array_almost_equal(window.numpy(), features[:W])

    def test_last_window_has_no_padding(self) -> None:
        n, W, F = 30, 8, 3
        rng = np.random.default_rng(3)
        features = rng.random((n, F)).astype(np.float32)
        labels = rng.integers(0, 2, n).astype(np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        window, _ = lazy[n - 1]
        np.testing.assert_array_almost_equal(window.numpy(), features[n - W : n])

    def test_label_equals_anchor_flow_label(self) -> None:
        n, W = 30, 8
        features = np.zeros((n, 2), dtype=np.float32)
        labels = np.arange(n, dtype=np.int64) % 2
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        for idx in range(len(lazy)):
            _, label = lazy[idx]
            assert label.item() == int(labels[idx])

    def test_label_train_stride_correct(self) -> None:
        n, W = 40, 8
        features = np.zeros((n, 2), dtype=np.float32)
        labels = np.arange(n, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=W // 2)
        expected_indices = list(range(0, n, W // 2))
        for idx, anchor in enumerate(expected_indices):
            _, label = lazy[idx]
            assert label.item() == anchor


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_n_features(self) -> None:
        features = np.zeros((20, 5), dtype=np.float32)
        labels = np.zeros(20, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, 4, stride=1)
        assert lazy.n_features == 5

    def test_n_classes_binary(self) -> None:
        features = np.zeros((20, 2), dtype=np.float32)
        labels = np.array([0, 1] * 10, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, 4, stride=1)
        assert lazy.n_classes == 2

    def test_window_size_property(self) -> None:
        features = np.zeros((20, 3), dtype=np.float32)
        labels = np.zeros(20, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, window_size=16, stride=1)
        assert lazy.window_size == 16

    def test_output_dtype_float32_and_long(self) -> None:
        features = np.zeros((20, 3), dtype=np.float32)
        labels = np.zeros(20, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, 8, stride=1)
        window, label = lazy[0]
        assert window.dtype == torch.float32
        assert label.dtype == torch.long

    def test_window_shape(self) -> None:
        n, W, F = 30, 8, 4
        features = np.zeros((n, F), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        window, _ = lazy[5]
        assert window.shape == (W, F)

    def test_len_test_stride_equals_n(self) -> None:
        n = 50
        features = np.zeros((n, 3), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, 8, stride=1)
        assert len(lazy) == n

    def test_len_train_stride(self) -> None:
        n, W = 60, 8
        features = np.zeros((n, 3), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=W // 2)
        assert len(lazy) == len(range(0, n, W // 2))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            LazySequenceFlowDataset(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                window_size=8,
                stride=1,
            )

    def test_single_record(self) -> None:
        W, F = 8, 3
        features = np.ones((1, F), dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        assert len(lazy) == 1
        window, label = lazy[0]
        assert window.shape == (W, F)
        np.testing.assert_array_equal(window[: W - 1].numpy(), np.zeros((W - 1, F)))
        np.testing.assert_array_equal(window[W - 1].numpy(), np.ones(F))
        assert label.item() == 1

    def test_window_size_equals_n(self) -> None:
        """Exactly W records: one window, first W-1 rows are padding."""
        W, F = 8, 2
        rng = np.random.default_rng(42)
        features = rng.random((W, F)).astype(np.float32)
        labels = np.zeros(W, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, W, stride=1)
        assert len(lazy) == W
        window, _ = lazy[0]
        np.testing.assert_array_equal(window[: W - 1].numpy(), np.zeros((W - 1, F)))
        np.testing.assert_array_almost_equal(window[W - 1].numpy(), features[0])

    def test_features_not_modified_in_place(self) -> None:
        features = np.ones((20, 3), dtype=np.float32)
        original = features.copy()
        labels = np.zeros(20, dtype=np.int64)
        lazy = LazySequenceFlowDataset(features, labels, 8, stride=1)
        _ = lazy[0]
        np.testing.assert_array_equal(features, original)

    def test_accepts_non_float32_features_and_casts(self) -> None:
        features = np.ones((20, 3), dtype=np.float64)
        labels = np.zeros(20, dtype=np.int32)
        lazy = LazySequenceFlowDataset(features, labels, 8, stride=1)
        window, label = lazy[0]
        assert window.dtype == torch.float32
        assert label.dtype == torch.long
