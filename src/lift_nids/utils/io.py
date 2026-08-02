"""Results persistence: save and load experiment outputs."""

from __future__ import annotations

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _coerce_json(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to JSON-serializable Python types.

    JSON keys must be str/int/float/bool/None — np.int64 is not accepted.
    This helper normalises the full results tree before json.dump.
    """
    if isinstance(obj, dict):
        return {
            (int(k) if isinstance(k, np.integer) else k): _coerce_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_coerce_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(
    results: dict[str, Any],
    output_path: Path,
    fmt: str = "json",
) -> None:
    """Save experiment results to disk.

    Args:
        results: Dict containing metrics, params, seed, timestamps, etc.
        output_path: Destination file path (.json or .parquet).
        fmt: "json" or "parquet".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_coerce_json(results), f, indent=2, default=str)
    elif fmt == "parquet":
        pd.DataFrame([results]).to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Choose 'json' or 'parquet'.")


def load_results(path: Path) -> dict[str, Any]:
    """Load experiment results from a JSON or Parquet file."""
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.iloc[0].to_dict()
    else:
        raise ValueError(f"Cannot infer format from suffix: {path.suffix!r}")


def _normalise_registry_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `horizon` to nullable string so numeric and categorical values
    written by different experiment runs can coexist in one Parquet file.

    Converts float-integers (1.0 → "1") and ints (1 → "1") while preserving
    strings like "cumulative". Missing values stay missing (not "nan"/"None").
    """
    if "horizon" not in df.columns:
        return df

    def _horizon_to_str(val: Any) -> Any:
        if pd.isna(val):
            return pd.NA
        if isinstance(val, (float, np.floating)) and val == int(val):
            return str(int(val))
        if isinstance(val, (int, np.integer)):
            return str(val)
        return str(val)

    df = df.copy()
    df["horizon"] = df["horizon"].map(_horizon_to_str).astype("string")
    return df


def append_results_registry(
    results: dict[str, Any],
    registry_dir: Path,
) -> None:
    """Append a result entry to the experiment registry Parquet.

    Creates `registry_dir/registry.parquet` if it does not exist, then
    appends the new row. Reads existing rows first to preserve history.

    The registry is deliberately append-only; so that re-runs of the same
    experiment remain distinguishable, each row is stamped with a unique
    ``run_id`` and a ``registered_at`` ISO timestamp unless the caller
    already provides them. Consumers wanting one row per logical experiment
    should select the latest ``registered_at`` per key.
    """
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "registry.parquet"

    stamped = dict(results)
    stamped.setdefault("run_id", uuid.uuid4().hex)
    stamped.setdefault(
        "registered_at", datetime.datetime.now().isoformat(timespec="seconds")
    )
    row = pd.DataFrame([stamped])
    if registry_path.exists():
        existing = pd.read_parquet(registry_path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined = _normalise_registry_schema(combined)
    tmp_path = registry_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, registry_path)
