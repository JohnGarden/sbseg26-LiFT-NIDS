"""Statistical validation: Bootstrap CI, Wilcoxon, Friedman + Nemenyi."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a metric vector.

    Args:
        scores: Array of per-fold or per-repetition metric values.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        confidence: CI confidence level (default 0.95).
        seed: Random seed.

    Returns:
        (lower, upper) bounds of the CI.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_means = np.array(
        [rng.choice(scores, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100.0 * alpha / 2))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2)))
    return lower, upper


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    n_comparisons: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float | bool]:
    """Wilcoxon signed-rank test with optional Bonferroni correction.

    Args:
        scores_a: Per-repetition scores for model A.
        scores_b: Per-repetition scores for model B.
        alpha: Significance level before correction.
        correction: "bonferroni" or "none".
        n_comparisons: Number of comparisons (used for Bonferroni).
        alternative: Hypothesis direction passed to scipy: "two-sided" (default),
            "greater" (scores_a > scores_b), or "less".  For a pre-specified
            directional comparison (e.g. LT > CNN-BiLSTM) use "greater"; at n=5
            this yields the minimum one-sided p=0.031 when all differences agree,
            whereas "two-sided" yields 0.0625.

    Returns:
        Dict with keys: statistic, p_value, p_adjusted, significant (bool).
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(
            f"scores_a and scores_b must have the same length; got {len(a)} vs {len(b)}"
        )
    try:
        stat, p_value = wilcoxon(a, b, alternative=alternative)
    except ValueError as exc:
        # scipy raises ValueError when all differences are zero
        raise ValueError(
            f"wilcoxon() failed (all differences may be zero): {exc}"
        ) from exc
    if correction == "bonferroni":
        p_adj = min(float(p_value) * n_comparisons, 1.0)
    else:
        p_adj = float(p_value)
    return {
        "statistic":   float(stat),
        "p_value":     float(p_value),
        "p_adjusted":  p_adj,
        "significant": bool(p_adj < alpha),
    }


# Nemenyi critical values q_alpha (from Demsar 2006, Table 5, alpha=0.05)
# Indexed by number of classifiers k (k=2..10)
_NEMENYI_Q: dict[int, float] = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi post-hoc test.

    CD = q_alpha * sqrt(k(k+1) / (6N))

    Only alpha=0.05 is supported via the tabulated values from Demsar 2006.
    For k > 10, the last tabulated value is reused (conservative approximation).
    """
    q = _NEMENYI_Q.get(min(k, 10), 3.164)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_nemenyi(
    scores: np.ndarray,
    model_names: list[str],
    alpha: float = 0.05,
) -> dict[str, object]:
    """Friedman test + Nemenyi post-hoc for multiple model comparison.

    Args:
        scores: Array of shape (n_blocks, n_models). Each row is one block —
                in classic Demsar usage a dataset/fold; in this project one
                random seed (see compute_axis2_stats.py) — and each column is
                one model. With seeds as blocks on a single dataset, results
                measure cross-seed rank stability, not multi-dataset
                superiority.
        model_names: List of model names matching columns of scores.
        alpha: Significance level (Nemenyi CD table only supports 0.05).

    Returns:
        Dict with keys:
          - friedman_stat (float)
          - p_value (float)
          - significant (bool) — Friedman test significant at alpha
          - avg_ranks (dict[str, float]) — average rank per model
          - critical_difference (float) — Nemenyi CD
          - nemenyi_significant (pd.DataFrame) — boolean k×k pairwise significance matrix
    """
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    if k != len(model_names):
        raise ValueError(f"scores has {k} columns but {len(model_names)} model_names")

    # Friedman test
    stat, p_value = friedmanchisquare(*[scores[:, j] for j in range(k)])

    # Average ranks (rank within each row, then average)
    ranks = np.apply_along_axis(
        lambda row: len(row) - row.argsort().argsort(),  # rank 1 = best (highest score)
        axis=1,
        arr=scores,
    )
    avg_ranks = {name: float(ranks[:, j].mean()) for j, name in enumerate(model_names)}

    # Nemenyi CD and pairwise significance
    cd = _nemenyi_cd(k, n, alpha=alpha)
    sig_matrix = pd.DataFrame(
        np.zeros((k, k), dtype=bool),
        index=model_names,
        columns=model_names,
    )
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i != j:
                sig_matrix.loc[name_i, name_j] = (
                    abs(avg_ranks[name_i] - avg_ranks[name_j]) > cd
                )

    return {
        "friedman_stat":       float(stat),
        "p_value":             float(p_value),
        "significant":         bool(p_value < alpha),
        "avg_ranks":           avg_ranks,
        "critical_difference": cd,
        "nemenyi_significant": sig_matrix,
    }
