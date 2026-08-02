"""MLP tabular baseline (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Multi-layer perceptron for tabular flow classification.

    Input:  (batch, n_features)
    Output: (batch, n_classes) — probabilities if output_activation is set, else logits

    Args:
        n_features: Number of input features.
        n_classes: Number of output classes.
        hidden_dims: Hidden layer sizes. Defaults to [256, 128, 64].
        dropout: Dropout probability.
        batch_norm: Whether to use BatchNorm after each linear layer.
        output_activation: "sigmoid" for binary (MAWIFlow), "softmax" for multiclass
            (CICIoT2023), or "none" for raw logits.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        dims = hidden_dims or [256, 128, 64]

        layers: list[nn.Module] = []
        in_dim = n_features
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

        if output_activation not in ("sigmoid", "softmax", "none"):
            raise ValueError(f"output_activation must be 'sigmoid', 'softmax', or 'none', got {output_activation!r}")
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(out)
        if self.output_activation == "softmax":
            return torch.softmax(out, dim=-1)
        return out

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
