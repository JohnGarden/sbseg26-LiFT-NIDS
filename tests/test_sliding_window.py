"""Tests for the materialized sliding-window test oracle (tests/_sliding_window_oracle.py).

The oracle is the reference implementation that ``LazySequenceFlowDataset`` is
checked against; these tests give its windowing arithmetic hand-computed coverage.
"""

import unittest

import numpy as np
import pandas as pd
from _sliding_window_oracle import (
    DEFAULT_WINDOW_SIZE,
    SlidingWindowBuilder,
    create_windows,
    windows_from_dataframe,
)


class TestCreateWindows(unittest.TestCase):
    def _make_data(self, n: int = 50, n_feat: int = 8) -> tuple:
        rng = np.random.default_rng(0)
        features = rng.random((n, n_feat)).astype(np.float32)
        labels = rng.integers(0, 2, size=n).astype(np.int64)
        return features, labels

    def test_output_shape_stride1(self) -> None:
        feats, labs = self._make_data(50, 8)
        windows, wlabs = create_windows(feats, labs, window_size=16, stride=1)
        self.assertEqual(windows.shape, (35, 16, 8))  # 50 - 16 + 1 = 35
        self.assertEqual(wlabs.shape, (35,))

    def test_output_shape_stride4(self) -> None:
        feats, labs = self._make_data(50, 8)
        windows, wlabs = create_windows(feats, labs, window_size=16, stride=4)
        expected = len(range(0, 50 - 16 + 1, 4))
        self.assertEqual(windows.shape[0], expected)

    def test_label_is_last_record(self) -> None:
        feats, labs = self._make_data(20, 4)
        windows, wlabs = create_windows(feats, labs, window_size=5, stride=1)
        # First window: records 0-4, label should be labs[4]
        self.assertEqual(wlabs[0], labs[4])
        # Second window: records 1-5, label should be labs[5]
        self.assertEqual(wlabs[1], labs[5])

    def test_default_window_size_constant(self) -> None:
        self.assertEqual(DEFAULT_WINDOW_SIZE, 16)

    def test_raises_when_too_short(self) -> None:
        feats, labs = self._make_data(5, 4)
        with self.assertRaises(ValueError):
            create_windows(feats, labs, window_size=16)

    def test_window_values_correct(self) -> None:
        feats = np.arange(30).reshape(10, 3).astype(np.float32)
        labs = np.zeros(10, dtype=np.int64)
        windows, _ = create_windows(feats, labs, window_size=3, stride=1)
        np.testing.assert_array_equal(windows[0], feats[:3])
        np.testing.assert_array_equal(windows[1], feats[1:4])


class TestWindowsFromDataframe(unittest.TestCase):
    def _make_df(self, n: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "t": range(n),
                "f1": rng.random(n),
                "f2": rng.random(n),
                "label": rng.integers(0, 2, n),
            }
        )

    def test_basic(self) -> None:
        df = self._make_df(40)
        windows, labels = windows_from_dataframe(
            df, ["f1", "f2"], "label", window_size=8, stride=1
        )
        self.assertEqual(windows.shape[1], 8)
        self.assertEqual(windows.shape[2], 2)

    def test_with_sort(self) -> None:
        df = self._make_df(20)
        df = df.sample(frac=1, random_state=0)  # shuffle
        windows, labels = windows_from_dataframe(
            df, ["f1", "f2"], "label", window_size=5, stride=1, sort_col="t"
        )
        self.assertEqual(windows.shape[0], 20 - 5 + 1)

    def test_grouped_windows(self) -> None:
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "group": ["A"] * 20 + ["B"] * 20,
                "f1": rng.random(40),
                "label": rng.integers(0, 2, 40),
            }
        )
        windows, labels = windows_from_dataframe(
            df, ["f1"], "label", window_size=5, stride=1, group_col="group"
        )
        # Each group of 20 produces 16 windows → 32 total
        self.assertEqual(windows.shape[0], 32)


class TestSlidingWindowBuilder(unittest.TestCase):
    """Tests for SlidingWindowBuilder."""

    _FEAT = ["f1", "f2", "f3"]
    _LABEL = "label"

    def _make_df(self, n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "f1": rng.random(n).astype(np.float32),
                "f2": rng.random(n).astype(np.float32),
                "f3": rng.random(n).astype(np.float32),
                "label": rng.integers(0, 2, n).astype(np.int64),
            }
        )

    # --- shape ---

    def test_train_shape(self) -> None:
        df = self._make_df(60)
        builder = SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=16)
        windows, labels = builder.build_train(df)
        expected_n = len(range(0, 60, 8))  # stride = W//2 = 8
        self.assertEqual(windows.shape, (expected_n, 16, 3))
        self.assertEqual(labels.shape, (expected_n,))

    def test_test_shape(self) -> None:
        df = self._make_df(60)
        builder = SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=16)
        windows, labels = builder.build_test(df)
        # stride=1: one window per flow
        self.assertEqual(windows.shape, (60, 16, 3))
        self.assertEqual(labels.shape, (60,))

    def test_window_size_8_and_32(self) -> None:
        df = self._make_df(64)
        for W in (8, 32):
            builder = SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=W)
            windows, _ = builder.build_test(df)
            self.assertEqual(windows.shape, (64, W, 3))

    def test_invalid_window_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=10)

    # --- label is the last flow ---

    def test_label_is_last_flow_test_mode(self) -> None:
        df = self._make_df(40)
        builder = SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=8)
        _, labels = builder.build_test(df)
        actual = df[self._LABEL].to_numpy(dtype=np.int64)
        # stride=1: window i covers flows [i-W+1:i+1], label = actual[i]
        np.testing.assert_array_equal(labels, actual)

    def test_label_is_last_flow_train_mode(self) -> None:
        df = self._make_df(40)
        builder = SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=8)
        _, labels = builder.build_train(df)
        actual = df[self._LABEL].to_numpy(dtype=np.int64)
        stride = 4  # W//2
        expected = np.array([actual[i] for i in range(0, 40, stride)], dtype=np.int64)
        np.testing.assert_array_equal(labels, expected)

    # --- no overlap between train and test splits ---

    def test_no_overlap_between_train_and_test(self) -> None:
        """Test windows from the test split contain no training-era feature values."""
        n = 50
        # Non-zero, unique per-row feature: train rows get values 1..25, test rows 26..50
        df = pd.DataFrame(
            {
                "f1": np.arange(1, n + 1, dtype=np.float32),
                "label": np.zeros(n, dtype=np.int64),
            }
        )
        split = 25
        train_df = df.iloc[:split].reset_index(drop=True)
        test_df = df.iloc[split:].reset_index(drop=True)

        builder = SlidingWindowBuilder(["f1"], "label", window_size=8)
        test_windows, _ = builder.build_test(test_df)

        # Values in test_df: 26.0 to 50.0.  Only 0.0 (padding) is allowed otherwise.
        for window in test_windows:
            for val in window[:, 0]:
                self.assertTrue(
                    val == 0.0 or val >= 26.0,
                    msg=f"Training-era value {val} leaked into test window",
                )

    # --- padding ---

    def test_padding_first_window(self) -> None:
        df = self._make_df(30)
        W = 8
        builder = SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=W)
        windows, _ = builder.build_test(df)

        # First window: W-1 all-zero rows, then the first data row
        first = windows[0]
        np.testing.assert_array_equal(first[: W - 1], np.zeros((W - 1, 3), dtype=np.float32))
        expected_last_row = df[self._FEAT].iloc[0].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(first[W - 1], expected_last_row)

    def test_padding_amount_decreases(self) -> None:
        df = self._make_df(20)
        W = 8
        builder = SlidingWindowBuilder(self._FEAT, self._LABEL, window_size=W)
        windows, _ = builder.build_test(df)

        # Window i should have max(0, W-1-i) zero rows at the start
        for i in range(min(W, len(windows))):
            n_zeros = W - 1 - i
            if n_zeros > 0:
                np.testing.assert_array_equal(
                    windows[i, :n_zeros],
                    np.zeros((n_zeros, 3), dtype=np.float32),
                )
            # Row at position n_zeros should be a data row (non-zero assumed from random data)
            data_row = df[self._FEAT].iloc[i].to_numpy(dtype=np.float32)
            np.testing.assert_array_equal(windows[i, W - 1], data_row)


if __name__ == "__main__":
    unittest.main()
