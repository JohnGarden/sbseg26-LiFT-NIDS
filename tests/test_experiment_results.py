"""Tests for the results-schema module: provenance, registry contract,
the save+register seam, and the per-axis record builders.
"""

from __future__ import annotations

import json

import pandas as pd

from lift_nids.experiments.results import (
    build_axis1_record,
    build_axis2_record,
    provenance,
    registry_row,
    write_result,
)


def test_provenance_stamps_timestamp():
    prov = provenance()
    assert set(prov) == {"timestamp"}
    # ISO-8601 to the second: parses cleanly.
    import datetime
    datetime.datetime.fromisoformat(prov["timestamp"])


def test_registry_row_keeps_scalars_drops_dicts_and_lists():
    record = {
        "model_family": "xgboost",
        "axis": 1,
        "elapsed_seconds": 12.3,
        "seeds": [42, 123],            # list -> dropped
        "metrics": {"f1_1": {"mean": 0.9}},  # dict -> dropped
        "best_params": {"max_depth": 8},     # dict -> dropped
    }
    row = registry_row(record)

    assert row == {"model_family": "xgboost", "axis": 1, "elapsed_seconds": 12.3}


def test_write_result_saves_json_and_appends_scalar_registry_row(tmp_path):
    record = {
        "model_family": "mlp",
        "axis": 1,
        "seeds": [42, 123],
        "metrics": {"f1_1": {"mean": 0.9}},
        "elapsed_seconds": 3.0,
    }
    results_path = tmp_path / "model" / "results.json"

    write_result(record, results_path, registry_dir=tmp_path)

    # Full record persisted to JSON.
    saved = json.loads(results_path.read_text())
    assert saved["metrics"] == {"f1_1": {"mean": 0.9}}
    assert saved["seeds"] == [42, 123]

    # Registry got only the scalar columns.
    reg = pd.read_parquet(tmp_path / "registry.parquet")
    assert set(reg.columns) >= {"model_family", "axis", "elapsed_seconds"}
    assert "metrics" not in reg.columns
    assert "seeds" not in reg.columns


def test_write_result_without_registry_dir_writes_no_registry(tmp_path):
    write_result({"model_family": "xgboost"}, tmp_path / "r.json")
    assert (tmp_path / "r.json").exists()
    assert not (tmp_path / "registry.parquet").exists()


def test_build_axis1_record_schema():
    rec = build_axis1_record(
        model_family="mlp",
        seeds=[42, 123],
        n_trials=15,
        num_workers=0,
        effective_batch_size=2048,
        best_hpo_val_f1=0.991,
        best_params={"lr": 1e-3},
        per_seed_metrics=[{"seed": 42, "f1_1": 0.98}],
        metrics={"f1_1": {"mean": 0.98}},
        elapsed_seconds=123.456,
    )

    assert rec["dataset"] == "CICIoT2023"
    assert rec["axis"] == 1
    assert rec["model_family"] == "mlp"
    assert rec["seeds"] == [42, 123]
    assert rec["effective_batch_size"] == 2048
    assert rec["best_hpo_val_f1"] == 0.991
    assert rec["metrics"] == {"f1_1": {"mean": 0.98}}
    assert rec["elapsed_seconds"] == 123.5  # rounded to 1 dp, as the runner did
    # provenance stamped
    assert "timestamp" in rec


def test_build_axis2_record_schema_and_derived_fields():
    rec = build_axis2_record(
        model_family="transformer",
        seeds=[42],
        horizon="cumulative",
        years=[2007, 2008, 2009],
        n_trials=5,
        sample_frac=None,
        run_mode="production",
        best_hpo_val_f1=0.8,
        best_params={"lr": 5e-4},
        effective_batch_size=512,
        per_seed_results=[{"seed": 42, "nAUT_1": 0.5}],
        metrics={"nAUT_1": {"mean": 0.5}},
        efficiency_summary={"n_params": 100},
        save_checkpoints=False,
        save_attention_artifacts=False,
        attention_samples_per_group=None,
        artifacts_dir=None,
        num_workers=0,
        xgboost_device="cpu",
        hpo_device="cuda",
        elapsed_seconds=9.99,
    )

    assert rec["dataset"] == "MAWIFlow"
    assert rec["axis"] == 2
    assert rec["horizon"] == "cumulative"
    # Fixed HPO-policy metadata the runner always wrote.
    assert rec["hpo_policy"] == "single_anchor_year"
    assert rec["hpo_split"] == "temporal_60_20_20"
    assert rec["hpo_reused_across_windows"] is True
    # anchor year derived from the first training year.
    assert rec["hpo_anchor_year"] == 2007
    assert rec["elapsed_seconds"] == 10.0
    assert "timestamp" in rec

    # A registry row of it keeps only scalars.
    row = registry_row(rec)
    assert "metrics" not in row and "per_seed_results" not in row and "years" not in row
    assert row["model_family"] == "transformer" and row["horizon"] == "cumulative"
