#!/usr/bin/env python3
"""Persist the Axis 2 statistical-test results as a reproducible artifact.

The Friedman/Nemenyi and Wilcoxon numbers reported in the paper (S4.2) were
previously reproducible only by hand from the per-seed JSONs. This script
reads ``<output-dir>/<model>/<window>/seed_<seed>.json`` and writes
``<output-dir>/stats_axis2.json`` containing:

* Friedman test + Nemenyi post-hoc (average ranks, critical difference,
  pairwise significance matrix) over the R=5 seed blocks;
* the single pre-specified confirmatory comparison -- one-sided Wilcoxon,
  LightTransformer > CNN-BiLSTM -- exactly as framed in the paper;
* the remaining pairwise comparisons, labelled exploratory, with
  Bonferroni-adjusted p-values reported for transparency only.

Seeds are used as blocks: this supports cross-seed rank-stability claims on a
single dataset, not Demsar-style multi-dataset superiority (see paper S5).

Usage:
    uv run python scripts/compute_axis2_stats.py
    uv run python scripts/compute_axis2_stats.py --output-dir data/results/axis2_final \\
        --window k_cumulative --metric nAUT_1
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lift_nids.evaluation.statistical_tests import friedman_nemenyi, wilcoxon_test
from lift_nids.experiments.constants import ALL_MODELS, SEEDS
# Pre-specified confirmatory comparison (paper S4.2): LT > CNN-BiLSTM.
PRIMARY_PAIR: tuple[str, str] = ("transformer", "cnn_bilstm")


def load_seed_scores(
    output_dir: Path,
    model: str,
    window: str,
    seeds: list[int],
    metric: str,
) -> list[float]:
    """Load one metric value per seed from ``<model>/<window>/seed_<seed>.json``.

    Args:
        output_dir: Axis 2 results root (e.g. ``data/results/axis2_final``).
        model: Model directory name.
        window: Training-window directory name (e.g. ``k_cumulative``).
        seeds: Seeds whose files must all be present.
        metric: Key to extract from each seed JSON (e.g. ``nAUT_1``).

    Returns:
        Metric values in the order of ``seeds``.

    Raises:
        FileNotFoundError: If any seed file is missing.
        KeyError: If the metric key is absent from a seed file.
    """
    scores: list[float] = []
    for seed in seeds:
        path = output_dir / model / window / f"seed_{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed artifact: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if metric not in data:
            raise KeyError(f"Metric {metric!r} not found in {path}")
        scores.append(float(data[metric]))
    return scores


def compute_stats(
    output_dir: Path,
    window: str = "k_cumulative",
    metric: str = "nAUT_1",
    models: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Compute Friedman/Nemenyi and Wilcoxon tests from per-seed artifacts.

    Args:
        output_dir: Axis 2 results root.
        window: Training-window directory name.
        metric: Per-seed metric to test.
        models: Model directory names (default: the four paper models).
        seeds: Seed list (default: the five paper seeds).

    Returns:
        JSON-serializable dict with the test results and provenance metadata.
    """
    models = models if models is not None else ALL_MODELS
    seeds = seeds if seeds is not None else SEEDS

    per_model = {m: load_seed_scores(output_dir, m, window, seeds, metric) for m in models}
    # Rows = seed blocks, columns = models.
    scores = np.array([[per_model[m][i] for m in models] for i in range(len(seeds))])

    friedman = friedman_nemenyi(scores, model_names=models)
    friedman_out = {
        "statistic": friedman["friedman_stat"],
        "p_value": friedman["p_value"],
        "significant": friedman["significant"],
        "avg_ranks": friedman["avg_ranks"],
        "nemenyi_critical_difference": friedman["critical_difference"],
        "nemenyi_significant": friedman["nemenyi_significant"].to_dict(),
    }

    n_pairs = len(models) * (len(models) - 1) // 2
    pairwise: list[dict] = []
    for a, b in itertools.combinations(models, 2):
        is_primary = {a, b} == set(PRIMARY_PAIR)
        # The primary pair tests its pre-specified direction (LT > CNN-BiLSTM);
        # exploratory pairs test the direction of the observed means.
        if is_primary:
            hi, lo = PRIMARY_PAIR
        else:
            hi, lo = (a, b) if np.mean(per_model[a]) >= np.mean(per_model[b]) else (b, a)
        test = wilcoxon_test(
            per_model[hi],
            per_model[lo],
            alternative="greater",
            correction="bonferroni",
            n_comparisons=n_pairs,
        )
        pairwise.append(
            {
                "greater": hi,
                "lesser": lo,
                "role": "primary_prespecified" if is_primary else "exploratory",
                "statistic": test["statistic"],
                "p_value": test["p_value"],
                "p_bonferroni_6": test["p_adjusted"],
            }
        )

    return {
        "axis": 2,
        "window": window,
        "metric": metric,
        "models": models,
        "seeds": seeds,
        "per_seed_scores": per_model,
        "friedman": friedman_out,
        "wilcoxon_one_sided": {
            "primary_prespecified": f"{PRIMARY_PAIR[0]} > {PRIMARY_PAIR[1]}",
            "note": (
                "Only the primary comparison is confirmatory; the remaining "
                "pairs are exploratory (direction chosen post hoc from the "
                "observed means). Bonferroni column adjusts for 6 pairs."
            ),
            "pairs": pairwise,
        },
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": f"{output_dir.as_posix()}/<model>/{window}/seed_<seed>.json",
        "script": Path(__file__).name,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/axis2_final"),
        help="Axis 2 results root (default: data/results/axis2_final)",
    )
    parser.add_argument("--window", default="k_cumulative")
    parser.add_argument("--metric", default="nAUT_1")
    args = parser.parse_args()

    stats = compute_stats(args.output_dir, window=args.window, metric=args.metric)

    out_path = args.output_dir / "stats_axis2.json"
    out_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    fr = stats["friedman"]
    primary = next(
        p for p in stats["wilcoxon_one_sided"]["pairs"] if p["role"] == "primary_prespecified"
    )
    print(f"Friedman: chi2={fr['statistic']:.4g}  p={fr['p_value']:.4g}")
    print(f"Avg ranks: {fr['avg_ranks']}")
    print(
        f"Primary Wilcoxon ({primary['greater']} > {primary['lesser']}): "
        f"p={primary['p_value']:.4g}"
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
