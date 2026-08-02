"""Generate the Axis 2 median F1+ drift trajectory figure (paper figure ``fig:trajectory``).

Reads k_cumulative results from data/results/axis2_final/, computes per-seed
median trajectories, and plots the cross-seed median with IQR band for Δt = 0..5.
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = Path("data/results/axis2_final")
OUT_DIR = Path("docs/paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "transformer": ("LightTransformer", "#27AE60", "D", "-"),
    "cnn_bilstm":  ("CNN-BiLSTM",       "#2980B9", "^", "--"),
    "xgboost":     ("XGBoost",          "#E67E22", "s", "-."),
    "mlp":         ("MLP",              "#C0392B", "o", ":"),
}

DT_MAX = 5


def load_trajectories(model_key: str) -> dict[int, list[float]]:
    """Return {dt: [f1+ values from each seed's median_trajectory]}."""
    path = BASE / model_key / "k_cumulative" / "results.json"
    with open(path) as f:
        data = json.load(f)

    by_dt: dict[int, list[float]] = {dt: [] for dt in range(DT_MAX + 1)}
    n_seeds = 0
    for seed_res in data.get("per_seed_results", []):
        mt = seed_res.get("median_trajectory", {})
        if not mt:
            continue
        n_seeds += 1
        for dt in range(DT_MAX + 1):
            val = mt.get(str(dt))
            if val is not None:
                by_dt[dt].append(val)

    print(f"  {model_key}: {n_seeds} seeds with trajectory data")
    return by_dt


def main():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for model_key, (label, color, marker, ls) in MODELS.items():
        by_dt = load_trajectories(model_key)

        dts, meds, q25s, q75s = [], [], [], []
        for dt in range(DT_MAX + 1):
            vals = by_dt[dt]
            if not vals:
                continue
            dts.append(dt)
            meds.append(statistics.median(vals))
            q25s.append(float(np.percentile(vals, 25)))
            q75s.append(float(np.percentile(vals, 75)))

        ax.plot(dts, meds, color=color, marker=marker, linestyle=ls,
                linewidth=2.2, markersize=6, label=label, zorder=3)
        ax.fill_between(dts, q25s, q75s, color=color, alpha=0.14, zorder=2)

    ax.set_xlabel(r"$\Delta t$ — years since training anchor", fontsize=11)
    ax.set_ylabel(r"$\widetilde{\mathrm{F1}}^+$ (median attack F1)", fontsize=11)
    ax.set_xlim(-0.25, DT_MAX + 0.25)
    ax.set_ylim(0.0, 0.88)
    ax.set_xticks(range(DT_MAX + 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9,
              edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout(pad=0.6)

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"trajectory-axis2.{ext}"
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
