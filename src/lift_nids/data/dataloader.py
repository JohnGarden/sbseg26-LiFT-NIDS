"""DataLoader factory for FlowDataset and LazySequenceFlowDataset."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Wrap a Dataset in a DataLoader with standard defaults.

    Args:
        dataset: FlowDataset or LazySequenceFlowDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Shuffle before each epoch (True for train, False for val/test).
        num_workers: Parallel data loading workers (0 = main process).
        pin_memory: Pin tensors to CUDA pinned memory (useful with GPU).
        drop_last: Drop the last incomplete batch.
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0).

    Returns:
        Configured DataLoader.
    """
    _persistent = persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=_persistent,
    )
