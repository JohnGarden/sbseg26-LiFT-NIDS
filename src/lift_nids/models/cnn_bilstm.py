"""CNN-BiLSTM sequential baseline (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """1-D CNN followed by a bidirectional LSTM for flow sequence classification.

    Input:  (batch, window_size, n_features)
    Output: (batch, n_classes) logits — prediction from last timestep
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        conv_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = conv_channels or [64, 128]
        kernels = kernel_sizes or [3, 3]
        assert len(channels) == len(kernels), "conv_channels and kernel_sizes must match"

        # 1-D Conv stack: input is (batch, n_features, seq_len) after transpose
        conv_blocks: list[nn.Module] = []
        in_ch = n_features
        for out_ch, k in zip(channels, kernels, strict=True):
            conv_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_blocks)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM doubles the hidden size
        self.classifier = nn.Linear(lstm_hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        # Conv1d expects (batch, channels, seq_len)
        x = self.conv(x.permute(0, 2, 1))   # -> (batch, out_ch, seq_len)
        x = x.permute(0, 2, 1)               # -> (batch, seq_len, out_ch)
        out, _ = self.lstm(x)                 # -> (batch, seq_len, hidden*2)
        last = out[:, -1, :]                  # Last timestep
        return self.classifier(self.dropout(last))
