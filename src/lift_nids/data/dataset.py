"""PyTorch Dataset classes for tabular and sequential flow data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    """Tabular dataset for single-flow models (XGBoost wrapper, MLP).

    Each sample is one flow record (feature vector + label).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(labels) == 0:
            raise ValueError(
                "Cannot create FlowDataset from empty arrays. "
                "Check that the split produced at least one sample."
            )
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.labels.max().item()) + 1


class LazySequenceFlowDataset(Dataset):
    """On-the-fly sequential dataset for window-based models (CNN-BiLSTM, Transformer).

    Avoids materialising the full (N, W, F) tensor — each window is built in
    ``__getitem__`` by slicing the base ``(N, F)`` array with zero-padding on
    the left for the first W-1 records.  Produces left-padded sliding windows
    with last-flow labelling (Last Token head convention).

    Args:
        features:    Float32 array of shape ``(N, F)``.
        labels:      Int64 array of shape ``(N,)``.
        window_size: Number of flows per window.
        stride:      Step between window anchors.
                     Use ``window_size // 2`` for training, ``1`` for test/val.
        left_pad:    Zero-pad windows near the start of the sequence (default True).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        stride: int,
        left_pad: bool = True,
    ) -> None:
        if len(labels) == 0:
            raise ValueError(
                "Cannot create LazySequenceFlowDataset from empty arrays. "
                "Check that the split produced at least one sample."
            )
        self._features = features.astype(np.float32)
        self._labels = labels.astype(np.int64)
        self.window_size = window_size
        self.stride = stride
        self.left_pad = left_pad
        # Pre-compute anchor positions: index i means the window ends at features[i].
        self._indices: list[int] = list(range(0, len(self._labels), stride))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self._indices[idx]
        W = self.window_size
        F = self._features.shape[1]
        start = i - W + 1  # may be negative → padding needed

        if start >= 0:
            window = torch.tensor(self._features[start : i + 1], dtype=torch.float32)
        else:
            n_pad = -start
            buf = np.zeros((W, F), dtype=np.float32)
            buf[n_pad:] = self._features[: i + 1]
            window = torch.tensor(buf, dtype=torch.float32)

        return window, torch.tensor(int(self._labels[i]), dtype=torch.long)

    @property
    def n_features(self) -> int:
        return self._features.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self._labels.max()) + 1
