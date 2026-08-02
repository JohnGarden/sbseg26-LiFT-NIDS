"""Early stopping criterion based on validation F1 score."""

from __future__ import annotations

import copy
from typing import Any


class EarlyStopping:
    """Stop training when validation F1 stops improving.

    Monitors a scalar metric (higher = better) with configurable patience
    and minimum delta. Optionally stores the best model state for restoration.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        restore_best: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self._best_score: float = -float("inf")
        self._best_epoch: int = 0
        self._counter: int = 0
        self._epoch: int = 0
        self._best_state: Any = None  # deep copy of model state_dict

    def step(self, metric: float, model: Any = None) -> bool:
        """Update state with current metric value.

        Args:
            metric: Validation metric (e.g. val_f1_macro). Higher is better.
            model: If restore_best=True, pass the model to snapshot its state.

        Returns:
            True if training should stop.
        """
        self._epoch += 1
        if metric > self._best_score + self.min_delta:
            self._best_score = metric
            self._best_epoch = self._epoch
            self._counter = 0
            if self.restore_best and model is not None:
                self._best_state = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1

        return self._counter >= self.patience

    def restore(self, model: Any) -> None:
        """Load the best observed model state into model (in-place)."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def improved(self) -> bool:
        """True if the last step() call was an improvement."""
        return self._counter == 0
