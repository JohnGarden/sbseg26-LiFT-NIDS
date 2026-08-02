"""Tests for src/lift_nids/data/dataset.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lift_nids.data.dataset import FlowDataset

RNG = np.random.default_rng(0)


def _features(n: int = 50, f: int = 10) -> np.ndarray:
    return RNG.standard_normal((n, f)).astype(np.float32)


def _labels(n: int = 50) -> np.ndarray:
    arr = RNG.integers(0, 2, size=n)
    arr[0], arr[1] = 0, 1  # guarantee both classes present
    return arr


class TestFlowDataset:
    def test_basic_len_and_shape(self) -> None:
        ds = FlowDataset(_features(50, 10), _labels(50))
        assert len(ds) == 50
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == ()

    def test_n_features(self) -> None:
        ds = FlowDataset(_features(20, 15), _labels(20))
        assert ds.n_features == 15

    def test_n_classes_binary(self) -> None:
        ds = FlowDataset(_features(20, 5), _labels(20))
        assert ds.n_classes == 2

    def test_dtype_float32_long(self) -> None:
        ds = FlowDataset(_features(10, 5), _labels(10))
        assert ds.features.dtype == torch.float32
        assert ds.labels.dtype == torch.long

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            FlowDataset(
                np.empty((0, 10), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

    def test_indexing_last_element(self) -> None:
        n = 30
        ds = FlowDataset(_features(n), _labels(n))
        x, y = ds[n - 1]
        assert x.shape == (10,)

    def test_labels_are_correct_values(self) -> None:
        y_np = np.array([0, 1, 0, 1], dtype=np.int64)
        X_np = np.zeros((4, 5), dtype=np.float32)
        ds = FlowDataset(X_np, y_np)
        assert ds.labels[0].item() == 0
        assert ds.labels[1].item() == 1
