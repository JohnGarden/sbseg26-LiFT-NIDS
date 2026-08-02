#!/usr/bin/env python3
"""Aggregate Axis 2 results from per-model runs into a unified summary.json.

Use this after running each model in a separate process:

    uv run python scripts/run_axis2.py --models xgboost  --output-dir data/results/axis2_final --resume
    uv run python scripts/run_axis2.py --models mlp       --output-dir data/results/axis2_final --resume
    uv run python scripts/run_axis2.py --models cnn_bilstm --output-dir data/results/axis2_final --resume
    uv run python scripts/run_axis2.py --models transformer --output-dir data/results/axis2_final --resume
    uv run python scripts/aggregate_axis2_summaries.py --output-dir data/results/axis2_final

The aggregated summary is saved to ``<output-dir>/summary.json``.  It does NOT
pretend success when results are missing: ``is_full_temporal_matrix`` is False
and ``run_status`` is ``"incomplete"`` or ``"partial_failed"`` whenever any
expected combination is absent or errored.

Exit code 0 → completed; 1 → incomplete or partial failure.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lift_nids.experiments.constants import ALL_MODELS, SEEDS

ALL_HORIZONS: list[str] = ["1", "2", "3", "cumulative"]


def _load_results(path: Path) -> dict | None:
    """Return parsed JSON or None on parse/IO error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def aggregate(
    output_dir: Path,
    expected_models: list[str],
    expected_horizons: list[str],
    expected_seeds: list[int],
) -> dict:
    """Read per-model results.json files and produce an aggregated summary dict."""
    h_keys = [f"k_{h}" for h in expected_horizons]
    expected_combs = [f"{m}/{hk}" for m in expected_models for hk in h_keys]

    results: dict[str, dict] = {}
    combinations_executed: list[str] = []
    combinations_failed: list[str] = []
    combinations_missing: list[str] = []
    seeds_found: set[int] = set()

    for model in expected_models:
        results[model] = {}
        for h, hk in zip(expected_horizons, h_keys):
            comb = f"{model}/{hk}"
            results_path = output_dir / model / hk / "results.json"

            if not results_path.exists():
                combinations_missing.append(comb)
                continue

            data = _load_results(results_path)
            if data is None:
                combinations_failed.append(comb)
                results[model][hk] = {"error": "failed to parse results.json"}
                continue

            if "error" in data:
                combinations_failed.append(comb)
                results[model][hk] = {"error": data["error"]}
                continue

            combinations_executed.append(comb)
            if data.get("seeds"):
                seeds_found.update(data["seeds"])

            results[model][hk] = {
                "horizon":          data.get("horizon"),
                "metrics":          data.get("metrics", {}),
                "best_params":      data.get("best_params", {}),
                "efficiency_summary": data.get("efficiency_summary", {}),
                "elapsed_seconds":  data.get("elapsed_seconds"),
                "seeds":            data.get("seeds"),
                "n_trials":         data.get("n_trials"),
                "sample_frac":      data.get("sample_frac"),
                "run_mode":         data.get("run_mode"),
                "timestamp":        data.get("timestamp"),
            }

    n_expected = len(expected_combs)
    is_full = (
        len(combinations_executed) == n_expected
        and not combinations_failed
        and not combinations_missing
    )
    is_full_seed_set = sorted(seeds_found) == sorted(expected_seeds)

    if is_full:
        run_status = "completed"
    elif combinations_failed:
        run_status = "partial_failed"
    else:
        run_status = "incomplete"

    return {
        "axis":                    2,
        "dataset":                 "MAWIFlow",
        "models_expected":         expected_models,
        "horizons_expected":       expected_horizons,
        "seeds_expected":          expected_seeds,
        "seeds_found":             sorted(seeds_found),
        "n_combinations_expected": n_expected,
        "combinations_executed":   combinations_executed,
        "combinations_failed":     combinations_failed,
        "combinations_missing":    combinations_missing,
        "is_full_temporal_matrix": is_full,
        "is_full_seed_set":        is_full_seed_set,
        "run_status":              run_status,
        "aggregated_at":           datetime.datetime.now().isoformat(timespec="seconds"),
        "results":                 results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Axis 2 per-model results into a unified summary.json."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory that contains the per-model subdirectories (e.g. xgboost/, mlp/, ...).",
    )
    parser.add_argument(
        "--expected-models", nargs="+", default=ALL_MODELS,
        metavar="MODEL",
        help=f"Models expected to be present. Default: {ALL_MODELS}.",
    )
    parser.add_argument(
        "--expected-horizons", nargs="+", default=ALL_HORIZONS,
        metavar="HORIZON",
        help=f"Horizons expected for each model. Default: {ALL_HORIZONS}.",
    )
    parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=list(SEEDS),
        metavar="SEED",
        help=f"Seeds expected in each result. Default: {SEEDS}.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: output-dir does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    summary = aggregate(
        output_dir=output_dir,
        expected_models=args.expected_models,
        expected_horizons=args.expected_horizons,
        expected_seeds=args.expected_seeds,
    )

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    n_exp = summary["n_combinations_expected"]
    n_ok  = len(summary["combinations_executed"])
    n_fail = len(summary["combinations_failed"])
    n_miss = len(summary["combinations_missing"])

    print(f"Summary -> {out_path}")
    print(f"  Executed : {n_ok}/{n_exp}")
    if n_fail:
        print(f"  Failed   : {n_fail}  {summary['combinations_failed']}")
    if n_miss:
        print(f"  Missing  : {n_miss}  {summary['combinations_missing']}")
    print(f"  is_full_temporal_matrix : {summary['is_full_temporal_matrix']}")
    print(f"  is_full_seed_set        : {summary['is_full_seed_set']}")
    print(f"  run_status              : {summary['run_status']}")

    sys.exit(0 if summary["run_status"] == "completed" else 1)


if __name__ == "__main__":
    main()
