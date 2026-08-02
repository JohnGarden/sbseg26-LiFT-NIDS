#!/usr/bin/env python3
"""Generate the Axis 2 Critical Difference diagram (Nemenyi post-hoc).

Reads the persisted statistical-test artifact (``stats_axis2.json``, produced by
``scripts/compute_axis2_stats.py``) and renders the Demsar-style CD diagram to
``docs/paper/figures/cd-diagram-axis2.{pdf,png}``.

The figure is deferred future work in the submitted manuscript (Conclusion);
it is generated here so the camera-ready/extended/thesis version can include it
without re-running anything. At R=5 seeds and k=4 models the Nemenyi CD is
~2.10 rank points, so only LightTransformer vs MLP (rank diff 3.0) is
significant — visually consistent with the paper's caveat that pairwise
inference is limited by the five-seed budget.

Usage:
    uv run python scripts/plot_cd_diagram_axis2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lift_nids.evaluation.cd_diagram import plot_cd_diagram

DISPLAY_NAMES = {
    "transformer": "LightTransformer",
    "cnn_bilstm": "CNN-BiLSTM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/results/axis2_final/stats_axis2.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/paper/figures")
    )
    args = parser.parse_args()

    stats = json.loads(args.stats_path.read_text(encoding="utf-8"))
    fr = stats["friedman"]
    avg_ranks = {
        DISPLAY_NAMES.get(model, model): rank
        for model, rank in fr["avg_ranks"].items()
    }
    cd = fr["nemenyi_critical_difference"]
    title = (
        f"Nemenyi CD diagram — {stats['metric']}, {stats['window']} "
        f"(R={len(stats['seeds'])} seeds, $\\alpha$=0.05)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot_cd_diagram(
            avg_ranks,
            critical_difference=cd,
            title=title,
            output_path=args.output_dir / f"cd-diagram-axis2.{ext}",
        )

    significant = [
        (a, b)
        for a, row in fr["nemenyi_significant"].items()
        for b, sig in row.items()
        if sig and a < b
    ]
    print(f"Average ranks: {avg_ranks}")
    print(f"Nemenyi CD (alpha=0.05): {cd:.4f}")
    print(f"Significant pairs: {significant or 'none'}")
    print(f"Saved: {args.output_dir / 'cd-diagram-axis2.{pdf,png}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
