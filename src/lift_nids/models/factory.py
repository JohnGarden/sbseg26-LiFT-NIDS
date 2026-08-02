"""Construct PyTorch models from a plain hyperparameter dict.

This is the single construction path shared by two callers that were previously
divergent:

* Optuna HPO — ``_suggest_pytorch_model`` suggests values with ``trial.suggest_*``
  and then builds the model, and
* refit from ``best_params`` — the runners rebuilt the model by feeding the
  stored params back through a fake Optuna trial (``_MockTrial``).

Both now delegate the object construction to :func:`build_torch_model`, so the
architecture wiring lives in exactly one place and no fake trial is needed.

The preset hidden-dim / conv-channel tables live here (rather than in
``optuna_search``) because they are part of how a model is *built*, not part of
the search procedure — the Optuna objective imports them to size its
``suggest_int`` ranges.
"""

from __future__ import annotations

import torch.nn as nn

from lift_nids.models.cnn_bilstm import CNNBiLSTM
from lift_nids.models.mlp_baseline import MLPBaseline
from lift_nids.models.transformer_light import LightTransformer

# Preset hidden-dim configurations for the MLP search space.
MLP_HIDDEN_CONFIGS: list[list[int]] = [
    [256, 128, 64],
    [512, 256, 128],
    [128, 64],
    [256, 128],
    [512, 256],
]

# Preset conv-channel configurations for the CNN-BiLSTM search space.
CNN_CHANNEL_CONFIGS: list[list[int]] = [
    [64, 128],
    [32, 64],
    [128, 256],
    [64],
    [128],
]


def build_torch_model(
    model_type: str,
    params: dict,
    n_features: int,
    n_classes: int = 2,
) -> tuple[nn.Module, float, float, int]:
    """Build a PyTorch model of ``model_type`` from a hyperparameter dict.

    Args:
        model_type: One of ``"mlp"``, ``"cnn_bilstm"``, ``"transformer"``.
        params: Hyperparameters as stored by Optuna (``best_params``). Required
            keys per family match the search space defined for that family.
        n_features: Number of input features.
        n_classes: Number of output classes.

    Returns:
        ``(model, lr, weight_decay, batch_size)`` — the model plus the three
        training hyperparameters the caller needs to build the optimizer and
        (for pre-built loaders) knows the searched batch size.

    Raises:
        ValueError: If ``model_type`` is not a known PyTorch family.
    """
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    if model_type == "mlp":
        model: nn.Module = MLPBaseline(
            n_features=n_features,
            n_classes=n_classes,
            hidden_dims=MLP_HIDDEN_CONFIGS[params["hidden_config_idx"]],
            dropout=params["dropout"],
            batch_norm=params["batch_norm"],
        )
    elif model_type == "cnn_bilstm":
        channels = CNN_CHANNEL_CONFIGS[params["channel_config_idx"]]
        model = CNNBiLSTM(
            n_features=n_features,
            n_classes=n_classes,
            conv_channels=channels,
            kernel_sizes=[3] * len(channels),
            lstm_hidden_size=params["lstm_hidden_size"],
            dropout=params["dropout"],
        )
    elif model_type == "transformer":
        model = LightTransformer(
            n_features=n_features,
            n_classes=n_classes,
            dropout=params["dropout"],
            head_dropout=params["head_dropout"],
        )
    else:
        raise ValueError(f"Unknown model_type for PyTorch: {model_type!r}")

    return model, lr, weight_decay, batch_size
