"""Tests for src/lift_nids/utils/io.py and src/lift_nids/utils/logging.py."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from lift_nids.utils.io import append_results_registry, load_results, save_results
from lift_nids.utils.logging import log_experiment_result, setup_logger

_SAMPLE: dict = {
    "experiment_name": "exp_test",
    "test_f1_macro": 0.85,
    "test_auc_roc": 0.92,
    "elapsed_seconds": 42.0,
}


class TestSaveLoadResults:
    def test_json_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "results.json"
        save_results(_SAMPLE, path, fmt="json")
        loaded = load_results(path)
        assert loaded["experiment_name"] == "exp_test"
        assert abs(loaded["test_f1_macro"] - 0.85) < 1e-6

    def test_parquet_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "results.parquet"
        save_results(_SAMPLE, path, fmt="parquet")
        loaded = load_results(path)
        assert loaded["experiment_name"] == "exp_test"
        assert abs(loaded["test_auc_roc"] - 0.92) < 1e-6

    def test_creates_parent_dirs(self, tmp_path) -> None:
        path = tmp_path / "nested" / "dir" / "results.json"
        save_results(_SAMPLE, path)
        assert path.exists()

    def test_unknown_format_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            save_results(_SAMPLE, tmp_path / "r.csv", fmt="csv")

    def test_unknown_load_suffix_raises(self, tmp_path) -> None:
        path = tmp_path / "r.txt"
        path.write_text("data")
        with pytest.raises(ValueError, match="Cannot infer"):
            load_results(path)


class TestAppendRegistry:
    def test_creates_file_if_absent(self, tmp_path) -> None:
        append_results_registry(_SAMPLE, tmp_path)
        assert (tmp_path / "registry.parquet").exists()

    def test_first_row_correct(self, tmp_path) -> None:
        append_results_registry(_SAMPLE, tmp_path)
        df = pd.read_parquet(tmp_path / "registry.parquet")
        assert len(df) == 1
        assert df.iloc[0]["experiment_name"] == "exp_test"

    def test_accumulates_rows(self, tmp_path) -> None:
        append_results_registry(_SAMPLE, tmp_path)
        append_results_registry({**_SAMPLE, "experiment_name": "exp_002"}, tmp_path)
        df = pd.read_parquet(tmp_path / "registry.parquet")
        assert len(df) == 2
        assert set(df["experiment_name"]) == {"exp_test", "exp_002"}

    def test_atomic_write_preserves_existing_on_failure(self, tmp_path, monkeypatch) -> None:
        append_results_registry(_SAMPLE, tmp_path)
        original = pd.read_parquet(tmp_path / "registry.parquet").copy()

        import lift_nids.utils.io as io_module

        def fail_replace(src, dst):
            raise OSError("simulated disk full")

        monkeypatch.setattr(io_module.os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated"):
            append_results_registry({**_SAMPLE, "experiment_name": "exp_fail"}, tmp_path)

        restored = pd.read_parquet(tmp_path / "registry.parquet")
        assert len(restored) == len(original)

    def test_creates_registry_dir_if_absent(self, tmp_path) -> None:
        subdir = tmp_path / "results" / "subdir"
        append_results_registry(_SAMPLE, subdir)
        assert (subdir / "registry.parquet").exists()

    def test_horizon_mixed_numeric_and_cumulative(self, tmp_path) -> None:
        append_results_registry(
            {**_SAMPLE, "model_family": "xgboost", "horizon": 1}, tmp_path
        )
        append_results_registry(
            {**_SAMPLE, "model_family": "xgboost", "horizon": "cumulative"}, tmp_path
        )
        df = pd.read_parquet(tmp_path / "registry.parquet")
        assert len(df) == 2
        assert set(df["horizon"].dropna().astype(str)) == {"1", "cumulative"}

    def test_stamps_run_id_and_registered_at(self, tmp_path) -> None:
        append_results_registry(_SAMPLE, tmp_path)
        df = pd.read_parquet(tmp_path / "registry.parquet")
        row = df.iloc[0]
        assert isinstance(row["run_id"], str) and len(row["run_id"]) > 0
        pd.Timestamp(row["registered_at"])  # parseable timestamp

    def test_duplicate_payloads_get_distinct_run_ids(self, tmp_path) -> None:
        append_results_registry(_SAMPLE, tmp_path)
        append_results_registry(_SAMPLE, tmp_path)
        df = pd.read_parquet(tmp_path / "registry.parquet")
        assert len(df) == 2
        assert df["run_id"].nunique() == 2

    def test_preserves_caller_run_id_and_registered_at(self, tmp_path) -> None:
        append_results_registry(
            {**_SAMPLE, "run_id": "my-run", "registered_at": "2026-01-01T00:00:00"},
            tmp_path,
        )
        df = pd.read_parquet(tmp_path / "registry.parquet")
        assert df.iloc[0]["run_id"] == "my-run"
        assert df.iloc[0]["registered_at"] == "2026-01-01T00:00:00"

    def test_does_not_mutate_input_dict(self, tmp_path) -> None:
        payload = dict(_SAMPLE)
        append_results_registry(payload, tmp_path)
        assert "run_id" not in payload and "registered_at" not in payload


class TestSetupLogger:
    def test_returns_logger_instance(self) -> None:
        logger = setup_logger("test_lift_io_basic")
        assert isinstance(logger, logging.Logger)

    def test_idempotent_multiple_calls(self) -> None:
        logger1 = setup_logger("test_lift_io_idem")
        logger2 = setup_logger("test_lift_io_idem")
        assert logger1 is logger2

    def test_writes_to_log_file(self, tmp_path) -> None:
        log_path = tmp_path / "test.log"
        logger = setup_logger("test_lift_io_file", log_file=log_path)
        logger.info("hello from test_io")
        assert log_path.exists()
        assert "hello from test_io" in log_path.read_text(encoding="utf-8")

    def test_log_experiment_result_no_crash(self) -> None:
        logger = setup_logger("test_lift_io_exp")
        log_experiment_result(
            logger, "exp_001", {"f1_macro": 0.9, "auc_roc": 0.95}, {"lr": 0.01}
        )

    def test_log_experiment_result_without_params(self) -> None:
        logger = setup_logger("test_lift_io_exp_noparams")
        log_experiment_result(logger, "exp_002", {"f1_macro": 0.8})
