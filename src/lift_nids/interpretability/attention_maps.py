"""Attention weight extraction and visualization for LightTransformer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def extract_attention_weights(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer_idx: int = -1,
) -> np.ndarray:
    """Extract per-head attention weights from a LightTransformer forward pass.

    Places the model in eval mode, runs a no-grad forward pass with
    ``return_attention_weights=True``, and returns the selected layer's weights
    as a CPU NumPy array.  The model's original training mode is restored
    unconditionally.

    Args:
        model: Fitted LightTransformer instance.
        x: Input tensor ``(batch, seq_len, n_features)``.
        layer_idx: Decoder layer to extract from (0-based or negative; -1 = last).

    Returns:
        Attention weight array ``(batch, n_heads, seq_len, seq_len)`` on CPU.

    Raises:
        IndexError: If ``layer_idx`` is out of range for the number of layers.
        ValueError: If the model does not return per-head attention weights.
    """
    was_training = model.training
    try:
        model.eval()

        device = next(model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            output = model(x, return_attention_weights=True)

        if not isinstance(output, tuple) or len(output) < 2:
            raise ValueError(
                "Model did not return attention weights. "
                "Ensure the model supports return_attention_weights=True."
            )

        _, attn_weights_list = output

        if not attn_weights_list:
            raise ValueError("Model returned an empty attention-weights list.")

        n_layers = len(attn_weights_list)
        if not (-n_layers <= layer_idx < n_layers):
            raise IndexError(
                f"layer_idx={layer_idx} is out of range for a model with {n_layers} "
                f"layers. Valid range: [{-n_layers}, {n_layers - 1}]."
            )

        attn = attn_weights_list[layer_idx]

        if attn is None:
            raise ValueError(
                f"Attention weights for layer {layer_idx} are None. "
                "Ensure need_weights=True and average_attn_weights=False in the model."
            )

        if attn.ndim != 4:
            raise ValueError(
                f"Expected attention shape (batch, n_heads, seq_len, seq_len); "
                f"got {tuple(attn.shape)}. "
                "Set average_attn_weights=False in nn.MultiheadAttention."
            )

        return attn.cpu().numpy()

    finally:
        if was_training:
            model.train()


def plot_attention_heatmap(
    weights: np.ndarray,
    position_labels: list[str] | None = None,
    head_idx: int = 0,
    sample_idx: int = 0,
    output_path: Path | None = None,
) -> None:
    """Plot an attention weight heatmap for a single head.

    Accepts attention arrays in three formats:
      - ``(seq_len, seq_len)``: single matrix, used as-is.
      - ``(n_heads, seq_len, seq_len)``: ``head_idx`` selects the head.
      - ``(batch, n_heads, seq_len, seq_len)``: ``sample_idx`` and ``head_idx``
        select sample and head.

    Axes are labelled as sequence-position indices (flow tokens), **not** as
    feature names — attention is token-vs-token, not feature-vs-feature.
    ``position_labels`` may supply custom strings for the token positions.

    Colour scale is fixed to ``[0, 1]``.  The figure is closed after saving or
    before returning to avoid memory leaks in loop usage.

    Args:
        weights: Attention array (2-D, 3-D, or 4-D; see above).
        position_labels: Optional labels for each sequence position.
            Length must equal ``seq_len``.  Defaults to ``["0", "1", …]``.
        head_idx: Head to visualise (for 3-D or 4-D input).
        sample_idx: Batch sample to visualise (for 4-D input only).
        output_path: If provided, save the figure as a PNG to this path.

    Raises:
        ValueError: If ``weights`` is not 2-D, 3-D, or 4-D.
        IndexError: If ``head_idx`` or ``sample_idx`` are out of range.
    """
    if weights.ndim == 2:
        mat = weights
    elif weights.ndim == 3:
        n_heads, _, _ = weights.shape
        if not (0 <= head_idx < n_heads):
            raise IndexError(
                f"head_idx={head_idx} is out of range for {n_heads} heads."
            )
        mat = weights[head_idx]
    elif weights.ndim == 4:
        batch, n_heads, _, _ = weights.shape
        if not (0 <= sample_idx < batch):
            raise IndexError(
                f"sample_idx={sample_idx} is out of range for batch size {batch}."
            )
        if not (0 <= head_idx < n_heads):
            raise IndexError(
                f"head_idx={head_idx} is out of range for {n_heads} heads."
            )
        mat = weights[sample_idx, head_idx]
    else:
        raise ValueError(
            f"Expected 2D, 3D, or 4D weights array; got shape {weights.shape}."
        )

    import matplotlib.pyplot as plt  # lazy: lets callers set the backend before first import

    seq_len = mat.shape[0]
    labels = position_labels if position_labels is not None else [str(i) for i in range(seq_len)]

    cell = max(0.4, min(0.7, 6.0 / seq_len))
    fig_size = max(4.0, seq_len * cell)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(labels, rotation=90, fontsize=max(6, 10 - seq_len // 8))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - seq_len // 8))
    ax.set_xlabel("Key position (attended to)")
    ax.set_ylabel("Query position")

    title = f"Attention head {head_idx}"
    if weights.ndim == 4:
        title += f" / sample {sample_idx}"
    ax.set_title(title)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
