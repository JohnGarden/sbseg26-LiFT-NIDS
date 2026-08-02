"""Lightweight causal Transformer (IM09 / FlowTransformer configuration).

Architecture fixed per the FlowTransformer study (IM09):
  - Decoder (causal, GPT-style)
  - 2 layers, 2 attention heads
  - d_model = 128, d_ff = 512 (4×d_model)
  - Record-level projection (input encoding)
  - Sinusoidal positional encoding
  - Last Token classification head
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

# Fixed architecture constants (DO NOT CHANGE — IM09)
D_MODEL: int = 128
N_HEADS: int = 2
NUM_LAYERS: int = 2
FF_DIM: int = 512   # 4 × d_model per IM09
DROPOUT: float = 0.1


def _build_sinusoidal_pe(max_seq_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding table (Vaswani et al., 2017)."""
    position = torch.arange(max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_seq_len, d_model)


class _CausalDecoderLayer(nn.Module):
    """Single Pre-LN causal decoder layer with optional attention weight output."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Pre-LN self-attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_w = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=mask,
            need_weights=need_weights,
            average_attn_weights=False,  # per-head (batch, n_heads, S, S); ignored when need_weights=False
        )
        x = residual + self.drop1(attn_out)

        # Pre-LN feed-forward
        residual = x
        x = residual + self.drop2(self.ff(self.norm2(x)))

        return x, attn_w


class LightTransformer(nn.Module):
    """Causal decoder Transformer for flow sequence classification.

    Input:  (batch, window_size, n_features)
    Output: (batch, n_classes) logits — Last Token classification head

    Architecture is fixed to the optimal config from FlowTransformer (IM09):
      2 causal decoder layers, 2 attention heads, d_model=128, d_ff=512,
      record-level projection, sinusoidal positional encoding, Last Token head.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        max_seq_len: int = 256,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Record-level projection: maps each flow vector to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        self.register_buffer("pos_enc", _build_sinusoidal_pe(max_seq_len, d_model))

        # Stack of causal decoder layers
        self.layers = nn.ModuleList([
            _CausalDecoderLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Last Token classification head (MLP with hidden layer)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, n_classes),
        )

        self._max_seq_len = max_seq_len

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular bool mask — True means 'do not attend' (future positions)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Forward pass with causal mask.

        Args:
            x: (batch, seq_len, n_features)
            return_attention_weights: if True, also return per-layer attention weights

        Returns:
            logits: (batch, n_classes) from Last Token position.
            attention_weights (only when return_attention_weights=True):
                list of length num_layers, each (batch, n_heads, seq_len, seq_len).
        """
        batch, seq_len, _ = x.shape

        # Record-level projection + sinusoidal positional encoding
        h = self.input_proj(x) + self.pos_enc[:seq_len]  # (batch, seq_len, d_model)

        mask = self._causal_mask(seq_len, x.device)

        all_attn_weights: list[torch.Tensor | None] = []
        for layer in self.layers:
            h, attn_w = layer(h, mask=mask, need_weights=return_attention_weights)
            if return_attention_weights:
                all_attn_weights.append(attn_w)

        logits = self.head(h[:, -1, :])   # (batch, n_classes)

        if return_attention_weights:
            return logits, all_attn_weights
        return logits
