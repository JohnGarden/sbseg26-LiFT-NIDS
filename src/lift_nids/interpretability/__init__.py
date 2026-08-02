"""Interpretability subpackage: attention maps."""

from lift_nids.interpretability.attention_maps import (
    extract_attention_weights,
    plot_attention_heatmap,
)

__all__ = [
    "extract_attention_weights",
    "plot_attention_heatmap",
]
