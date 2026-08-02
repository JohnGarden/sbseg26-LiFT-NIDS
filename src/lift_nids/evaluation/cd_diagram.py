"""Critical Difference (CD) diagram for multi-model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _maximal_cliques(ranks: list[float], cd: float) -> list[tuple[int, int]]:
    """Index ranges (start, end) of maximal cliques among sorted ranks.

    A clique is a maximal run of models whose pairwise rank differences with
    the leftmost member are all <= ``cd``. Cliques fully contained in another
    clique are dropped (Demsar 2006 drawing convention).
    """
    k = len(ranks)
    intervals = []
    for i in range(k):
        end = i
        for j in range(i + 1, k):
            if abs(ranks[j] - ranks[i]) <= cd:
                end = j
        if end > i:
            intervals.append((i, end))
    return [
        iv
        for iv in intervals
        if not any(o != iv and o[0] <= iv[0] and iv[1] <= o[1] for o in intervals)
    ]


def plot_cd_diagram(
    avg_ranks: dict[str, float],
    critical_difference: float,
    title: str = "Critical Difference Diagram",
    output_path: Path | None = None,
) -> None:
    """Plot a Critical Difference diagram (Demsar 2006 style).

    Models are placed on a horizontal axis by average rank (left = better rank).
    A horizontal bar of length CD is drawn at the top to indicate non-significant
    differences. Cliques (groups of models not significantly different from each
    other) are marked with vertical lines connecting them.

    Args:
        avg_ranks: Dict mapping model name -> average rank (lower rank = better).
        critical_difference: CD value from Nemenyi post-hoc test.
        title: Plot title.
        output_path: If provided, save figure to this path (PNG/PDF).
    """
    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
    names  = [m for m, _ in sorted_models]
    ranks  = [r for _, r in sorted_models]
    k      = len(names)
    max_rank = max(ranks)

    fig, ax = plt.subplots(figsize=(max(6, k * 1.5), 3))
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Horizontal rank axis
    ax.plot([0.5, max_rank + 0.5], [1.0, 1.0], color="black", lw=1.5)
    for r in np.arange(1, max_rank + 1):
        ax.plot([r, r], [0.95, 1.05], color="black", lw=1.2)
        ax.text(r, 0.88, str(int(r)), ha="center", va="top", fontsize=8)

    # Model labels and tick marks
    for i, (name, rank) in enumerate(zip(names, ranks, strict=True)):
        y_text = 1.55 if i % 2 == 0 else 1.28
        ax.plot([rank, rank], [1.0, y_text - 0.05], color="steelblue", lw=1, ls="--")
        ax.text(rank, y_text, name, ha="center", va="bottom", fontsize=9,
                color="steelblue", fontweight="bold")

    # CD bar at top-left
    cd_x_start = 0.6
    cd_x_end   = cd_x_start + critical_difference
    ax.annotate(
        "", xy=(cd_x_end, 1.85), xytext=(cd_x_start, 1.85),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax.text(
        (cd_x_start + cd_x_end) / 2, 1.9,
        f"CD = {critical_difference:.3f}",
        ha="center", va="bottom", fontsize=8,
    )

    # Clique bars: maximal groups of models not significantly different
    # (rank diff <= CD), staggered vertically so overlapping cliques stay
    # visually distinct (Demsar 2006 style).
    for level, (start, end) in enumerate(_maximal_cliques(ranks, critical_difference)):
        y = 0.74 - 0.10 * level
        ax.plot(
            [ranks[start], ranks[end]],
            [y, y],
            color="black", lw=3, solid_capstyle="round",
        )

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
