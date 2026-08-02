"""Utilities for deterministic and traceable experiment execution."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed the RNGs and select the cuDNN backend mode for LiFT-NIDS.

    This is the single owner of cuDNN determinism configuration — no caller
    should set ``torch.backends.cudnn.*`` directly.

    Args:
        seed: Seed applied to ``PYTHONHASHSEED``, ``random``, NumPy, and torch.
        deterministic: When True (default), force deterministic cuDNN algorithms
            (``cudnn.deterministic=True``, ``cudnn.benchmark=False``) so runs are
            reproducible — the protocol the project requires. When False, allow
            the cuDNN autotuner (``cudnn.deterministic=False``,
            ``cudnn.benchmark=True``) for speed, at the cost of run-to-run
            reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    except ImportError:
        pass
