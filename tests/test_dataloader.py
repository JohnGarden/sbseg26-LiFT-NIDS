"""Tests for src/lift_nids/data/dataloader.py."""

from __future__ import annotations

import numpy as np
import torch

from lift_nids.data.dataloader import build_dataloader
from lift_nids.data.dataset import FlowDataset, LazySequenceFlowDataset

RNG = np.random.default_rng(0)


def _tabular_dataset(n: int = 100, f: int = 10) -> FlowDataset:
    X = RNG.standard_normal((n, f)).astype(np.float32)
    y = RNG.integers(0, 2, size=n)
    return FlowDataset(X, y)


def _seq_dataset(n: int = 100, w: int = 8, f: int = 10) -> LazySequenceFlowDataset:
    # LazySequenceFlowDataset builds (w, f) windows from a base (n, f) flow array,
    # with stride=1 producing exactly n windows.
    X = RNG.standard_normal((n, f)).astype(np.float32)
    y = RNG.integers(0, 2, size=n)
    return LazySequenceFlowDataset(X, y, window_size=w, stride=1)


class TestBuildDataloader:
    def test_tabular_batch_shape(self) -> None:
        dl = build_dataloader(_tabular_dataset(n=100, f=10), batch_size=32)
        X_batch, y_batch = next(iter(dl))
        assert X_batch.shape == (32, 10)
        assert y_batch.shape == (32,)

    def test_sequence_batch_shape(self) -> None:
        dl = build_dataloader(_seq_dataset(n=64, w=8, f=10), batch_size=16)
        X_batch, y_batch = next(iter(dl))
        assert X_batch.shape == (16, 8, 10)
        assert y_batch.shape == (16,)

    def test_pin_memory_false_cpu(self) -> None:
        dl = build_dataloader(_tabular_dataset(), batch_size=32, pin_memory=False)
        assert dl.pin_memory is False
        X_batch, _ = next(iter(dl))
        assert not X_batch.is_pinned()

    def test_shuffle_no_error(self) -> None:
        dl = build_dataloader(_tabular_dataset(n=100), batch_size=32, shuffle=True)
        X_batch, y_batch = next(iter(dl))
        assert X_batch.shape == (32, 10)
        assert y_batch.shape == (32,)

    def test_drop_last_reduces_batch_count(self) -> None:
        dl_keep = build_dataloader(_tabular_dataset(n=70), batch_size=32, drop_last=False)
        dl_drop = build_dataloader(_tabular_dataset(n=70), batch_size=32, drop_last=True)
        assert sum(1 for _ in dl_keep) > sum(1 for _ in dl_drop)

    def test_batch_size_one(self) -> None:
        dl = build_dataloader(_tabular_dataset(n=5), batch_size=1)
        batches = list(dl)
        assert len(batches) == 5
        assert batches[0][0].shape == (1, 10)

    def test_tensor_dtype(self) -> None:
        dl = build_dataloader(_tabular_dataset(), batch_size=16)
        X_batch, y_batch = next(iter(dl))
        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.long
