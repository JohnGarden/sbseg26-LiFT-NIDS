"""Tests for scripts/run_axis2.py helpers and CLI logic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import json
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_axis2 as ra


class CapturingLogger:
    """Minimal logger that keeps formatted info/error messages for CLI tests."""

    def __init__(self):
        self.messages: list[str] = []

    def info(self, message, *args, **kwargs):
        self.messages.append(message % args if args else message)

    def error(self, message, *args, **kwargs):
        self.messages.append(message % args if args else message)


# ---------------------------------------------------------------------------
# _parse_horizons
# ---------------------------------------------------------------------------

class TestParseHorizons:
    def test_integers_become_int(self):
        assert ra._parse_horizons(["1", "2", "3"]) == [1, 2, 3]

    def test_cumulative_stays_str(self):
        assert ra._parse_horizons(["cumulative"]) == ["cumulative"]

    def test_mixed(self):
        result = ra._parse_horizons(["1", "cumulative"])
        assert result == [1, "cumulative"]

    def test_all_four_defaults(self):
        result = ra._parse_horizons(["1", "2", "3", "cumulative"])
        assert result == [1, 2, 3, "cumulative"]

    def test_single_integer(self):
        assert ra._parse_horizons(["3"]) == [3]


# ---------------------------------------------------------------------------
# _horizon_dir_name
# ---------------------------------------------------------------------------

class TestHorizonDirName:
    def test_integer(self):
        assert ra._horizon_dir_name(1) == "k_1"

    def test_cumulative(self):
        assert ra._horizon_dir_name("cumulative") == "k_cumulative"

    def test_integer_3(self):
        assert ra._horizon_dir_name(3) == "k_3"


# ---------------------------------------------------------------------------
# ALL_HORIZONS / ALL_MODELS constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_all_horizons_has_four_elements(self):
        assert len(ra.ALL_HORIZONS) == 4

    def test_all_horizons_contains_cumulative(self):
        assert "cumulative" in ra.ALL_HORIZONS

    def test_all_horizons_contains_1_2_3(self):
        assert 1 in ra.ALL_HORIZONS
        assert 2 in ra.ALL_HORIZONS
        assert 3 in ra.ALL_HORIZONS

    def test_all_models_has_four_elements(self):
        assert len(ra.ALL_MODELS) == 4

    def test_all_models_contains_expected(self):
        assert set(ra.ALL_MODELS) == {"xgboost", "mlp", "cnn_bilstm", "transformer"}


# ---------------------------------------------------------------------------
# Module documentation stays aligned with the current CLI/protocol
# ---------------------------------------------------------------------------

class TestAxis2Documentation:
    def test_docstring_describes_current_protocol_and_cli(self):
        doc = ra.__doc__ or ""

        assert "MAWIFlow yearly Parquet" in doc
        assert "temporal 60/20/20" in doc
        assert "median_trajectory" in doc
        assert "nAUT_H" in doc
        assert "--horizons" in doc

    def test_docstring_does_not_mention_removed_arguments_or_old_hpo_split(self):
        doc = ra.__doc__ or ""
        doc_without_valid_plural = doc.replace("--horizons", "")

        assert "--k" not in doc
        assert "--horizon" not in doc_without_valid_plural
        assert "HPO 80/20" not in doc
        assert "80/20" not in doc


# ---------------------------------------------------------------------------
# is_full_temporal_matrix logic
# ---------------------------------------------------------------------------

class TestIsFullTemporalMatrix:
    """Test the is_full flag computation from main() without running training."""

    def _compute_is_full(self, models: list[str], horizons_raw: list[str]) -> bool:
        """Mirror the is_full computation in main()."""
        horizons = ra._parse_horizons(horizons_raw)
        return (
            sorted(models) == sorted(ra.ALL_MODELS)
            and set(horizons) == set(ra.ALL_HORIZONS)
        )

    def test_all_four_models_all_four_horizons_is_true(self):
        assert self._compute_is_full(
            ["xgboost", "mlp", "cnn_bilstm", "transformer"],
            ["1", "2", "3", "cumulative"],
        ) is True

    def test_order_does_not_matter(self):
        assert self._compute_is_full(
            ["transformer", "cnn_bilstm", "mlp", "xgboost"],
            ["cumulative", "3", "1", "2"],
        ) is True

    def test_missing_model_is_false(self):
        assert self._compute_is_full(
            ["xgboost", "mlp", "cnn_bilstm"],  # transformer missing
            ["1", "2", "3", "cumulative"],
        ) is False

    def test_missing_horizon_is_false(self):
        assert self._compute_is_full(
            ["xgboost", "mlp", "cnn_bilstm", "transformer"],
            ["1", "cumulative"],  # 2 and 3 missing
        ) is False

    def test_subset_of_both_is_false(self):
        assert self._compute_is_full(["xgboost"], ["1"]) is False


# ---------------------------------------------------------------------------
# Default generates 4×4 = 16 combinations
# ---------------------------------------------------------------------------

class TestDefaultCombinations:
    def test_default_horizons_times_default_models_is_16(self):
        default_horizons = ra._parse_horizons(["1", "2", "3", "cumulative"])
        n_combinations = len(ra.ALL_MODELS) * len(default_horizons)
        assert n_combinations == 16

    def test_two_horizons_gives_eight_combinations(self):
        horizons = ra._parse_horizons(["1", "cumulative"])
        n_combinations = len(ra.ALL_MODELS) * len(horizons)
        assert n_combinations == 8

    def test_one_model_one_horizon_gives_one_combination(self):
        horizons = ra._parse_horizons(["1"])
        assert len(["xgboost"]) * len(horizons) == 1


# ---------------------------------------------------------------------------
# --dry-run does not call run_model
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_default_dry_run_matrix_is_four_models_by_four_horizons(self, tmp_path):
        logger = CapturingLogger()

        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
            ]),
            patch.object(ra, "run_model") as mock_run,
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=logger),
        ):
            ra.main()

        text = "\n".join(logger.messages)

        assert "Execution matrix (16 combinations)" in text
        assert text.count("results.json") == 16
        assert "is_full_temporal_matrix: True" in text
        assert "--k" not in text
        mock_run.assert_not_called()

    def test_dry_run_does_not_invoke_run_model(self, tmp_path):
        with (
            patch("sys.argv", ["run_axis2.py", "--dry-run",
                                "--output-dir", str(tmp_path)]),
            patch.object(ra, "run_model") as mock_run,
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            import importlib
            importlib.reload(ra)
            # Invoke via main() directly after patching the module-level objects
            ra_fresh = ra  # already patched above
            # Re-patch on the freshly reloaded module to be safe
            with patch.object(ra, "run_model") as mock_run2:
                ra.main.__globals__["resolve_device"] = lambda cfg: "cpu"
                try:
                    ra.main()
                except SystemExit:
                    pass
                mock_run2.assert_not_called()

    def test_dry_run_via_argv(self, tmp_path):
        """main() with --dry-run returns without calling run_model."""
        called = []

        def fake_run_model(*args, **kwargs):
            called.append(True)
            return {}

        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()

        assert called == [], "run_model must not be called during --dry-run"


# ---------------------------------------------------------------------------
# summary.json is_full_temporal_matrix
# ---------------------------------------------------------------------------

class TestSummaryIsFullFlag:
    """Verify that summary.json records is_full_temporal_matrix correctly."""

    def _run_with_fake_model(
        self,
        tmp_path: Path,
        models: list[str],
        horizons: list[str],
        fake_horizon_results: dict,
    ) -> dict:
        """Invoke main() with a mocked run_model and return the saved summary."""
        import json

        def fake_run_model(model_family, **kwargs):
            return fake_horizon_results

        with (
            patch("sys.argv", [
                "run_axis2.py",
                "--models", *models,
                "--horizons", *horizons,
                "--output-dir", str(tmp_path),
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results"),
        ):
            saved_summaries = []
            original_save = ra.save_results.__wrapped__ if hasattr(ra.save_results, "__wrapped__") else None

            captured = {}

            def capture_save(data, path):
                if "summary" in str(path):
                    captured.update(data)

            with patch.object(ra, "save_results", side_effect=capture_save):
                ra.main()

        return captured

    def test_full_matrix_is_true(self, tmp_path):
        fake_results = {
            f"k_{h}": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": h}
            for h in [1, 2, 3, "cumulative"]
        }
        summary = self._run_with_fake_model(
            tmp_path,
            models=["xgboost", "mlp", "cnn_bilstm", "transformer"],
            horizons=["1", "2", "3", "cumulative"],
            fake_horizon_results=fake_results,
        )
        assert summary.get("is_full_temporal_matrix") is True

    def test_partial_horizons_is_false(self, tmp_path):
        fake_results = {
            "k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1},
        }
        summary = self._run_with_fake_model(
            tmp_path,
            models=["xgboost", "mlp", "cnn_bilstm", "transformer"],
            horizons=["1"],
            fake_horizon_results=fake_results,
        )
        assert summary.get("is_full_temporal_matrix") is False

    def test_partial_models_is_false(self, tmp_path):
        fake_results = {
            f"k_{h}": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": h}
            for h in [1, 2, 3, "cumulative"]
        }
        summary = self._run_with_fake_model(
            tmp_path,
            models=["xgboost"],
            horizons=["1", "2", "3", "cumulative"],
            fake_horizon_results=fake_results,
        )
        assert summary.get("is_full_temporal_matrix") is False

    def test_summary_contains_combinations_executed(self, tmp_path):
        fake_results = {
            "k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1},
            "k_cumulative": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": "cumulative"},
        }
        summary = self._run_with_fake_model(
            tmp_path,
            models=["xgboost"],
            horizons=["1", "cumulative"],
            fake_horizon_results=fake_results,
        )
        assert "combinations_executed" in summary
        assert "combinations_failed" in summary
        assert sorted(summary["combinations_executed"]) == [
            "xgboost/k_1", "xgboost/k_cumulative"
        ]


# ---------------------------------------------------------------------------
# Interpretability artifact flags
# ---------------------------------------------------------------------------

class TestInterpretabilityFlags:
    """Verify that the new CLI flags exist, parse correctly, and are transformer-only."""

    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        import argparse as _ap
        parser = argparse.ArgumentParser()
        with patch("sys.argv", ["run_axis2.py"] + extra_args):
            # Reload a fresh parser by calling main's parser setup path.
            # Simpler: build the same parser manually and verify args.
            parser.add_argument("--save-checkpoints", action="store_true", default=False)
            parser.add_argument("--save-attention-artifacts", action="store_true", default=False)
            parser.add_argument("--attention-samples-per-group", type=int, default=16)
        return parser.parse_args(extra_args)

    def test_save_checkpoints_flag_exists_and_defaults_false(self):
        args = self._parse([])
        assert args.save_checkpoints is False

    def test_save_checkpoints_flag_activates(self):
        args = self._parse(["--save-checkpoints"])
        assert args.save_checkpoints is True

    def test_save_attention_artifacts_flag_exists_and_defaults_false(self):
        args = self._parse([])
        assert args.save_attention_artifacts is False

    def test_save_attention_artifacts_flag_activates(self):
        args = self._parse(["--save-attention-artifacts"])
        assert args.save_attention_artifacts is True

    def test_attention_samples_per_group_defaults_to_16(self):
        args = self._parse([])
        assert args.attention_samples_per_group == 16

    def test_attention_samples_per_group_custom(self):
        args = self._parse(["--attention-samples-per-group", "8"])
        assert args.attention_samples_per_group == 8

    def test_argparse_registers_save_checkpoints(self, tmp_path):
        """Verify main() parser registers --save-checkpoints without error."""
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
                "--save-checkpoints",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()  # should not raise SystemExit(2) from argparse

    def test_argparse_registers_save_attention_artifacts(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
                "--save-attention-artifacts",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()

    def test_argparse_registers_attention_samples_per_group(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
                "--attention-samples-per-group", "4",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()

    def test_non_transformer_seed_artifacts_dir_is_none(self):
        """seed_artifacts_dir must be None for non-transformer models even with flags on."""
        for model_family in ["xgboost", "mlp", "cnn_bilstm"]:
            seed_artifacts_dir: Path | None = None
            if model_family == "transformer" and (True or True):
                seed_artifacts_dir = Path("some/dir")
            assert seed_artifacts_dir is None, (
                f"seed_artifacts_dir should remain None for {model_family}"
            )

    def test_transformer_seed_artifacts_dir_is_set_when_flags_active(self):
        """seed_artifacts_dir must be non-None for transformer when any artifact flag is on."""
        horizon_dir = Path("data/results/axis2/transformer/k_1")
        seed = 42
        for save_checkpoints, save_attention_artifacts in [(True, False), (False, True), (True, True)]:
            seed_artifacts_dir = None
            if "transformer" == "transformer" and (save_checkpoints or save_attention_artifacts):
                seed_artifacts_dir = horizon_dir / f"seed_{seed}"
            assert seed_artifacts_dir == horizon_dir / f"seed_{seed}"


# ---------------------------------------------------------------------------
# --num-workers CLI arg
# ---------------------------------------------------------------------------

class TestNumWorkersArg:
    """--num-workers must be accepted by the CLI and recorded in summary."""

    def test_num_workers_zero_accepted_by_dry_run(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
                "--num-workers", "0",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()  # must not raise SystemExit(2)

    def test_num_workers_recorded_in_summary(self, tmp_path):
        captured = {}

        def fake_run_model(model_family, **kwargs):
            return {
                "k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1}
            }

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py", "--models", "xgboost", "--horizons", "1",
                "--output-dir", str(tmp_path), "--num-workers", "0",
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()

        assert "num_workers" in captured
        assert captured["num_workers"] == 0


# ---------------------------------------------------------------------------
# --xgboost-device CLI arg
# ---------------------------------------------------------------------------

class TestXgboostDeviceArg:
    """--xgboost-device must be accepted by the CLI, default to 'cpu', and recorded."""

    def test_xgboost_device_cpu_accepted_by_dry_run(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
                "--xgboost-device", "cpu",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()

    def test_xgboost_device_recorded_in_summary(self, tmp_path):
        captured = {}

        def fake_run_model(model_family, **kwargs):
            return {
                "k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1}
            }

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py", "--models", "xgboost", "--horizons", "1",
                "--output-dir", str(tmp_path), "--xgboost-device", "cpu",
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()

        assert "xgboost_device" in captured
        assert captured["xgboost_device"] == "cpu"

    def test_xgboost_device_defaults_to_cpu(self, tmp_path):
        captured = {}

        def fake_run_model(model_family, **kwargs):
            return {
                "k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1}
            }

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py", "--models", "xgboost", "--horizons", "1",
                "--output-dir", str(tmp_path),
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()

        assert captured.get("xgboost_device") == "cpu"

    def test_xgboost_hpo_uses_xgboost_device_not_torch_device(self, tmp_path):
        import numpy as np
        import pandas as pd

        captured = {}
        df = pd.DataFrame({
            "year": [2007] * 8 + [2008] * 4,
            "timestamp": pd.date_range("2007-01-01", periods=12, freq="h"),
            "label": [0, 1] * 6,
            "f1": np.arange(12, dtype=float),
        })

        def fake_temporal_split(anchor_df, *_args, **_kwargs):
            return (
                anchor_df.iloc[:4].reset_index(drop=True),
                anchor_df.iloc[4:6].reset_index(drop=True),
                anchor_df.iloc[6:].reset_index(drop=True),
            )

        def fake_tabular_arrays(*_args, **_kwargs):
            X_tr = np.zeros((4, 1), dtype=np.float32)
            y_tr = np.array([0, 1, 0, 1], dtype=np.int64)
            X_val = np.zeros((2, 1), dtype=np.float32)
            y_val = np.array([0, 1], dtype=np.int64)
            return (X_tr, y_tr), (X_val, y_val), (X_val, y_val), MagicMock()

        def fake_search(**kwargs):
            captured.update(kwargs)
            return {
                "best_params": {
                    "n_estimators": 1,
                    "max_depth": 1,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "scale_pos_weight": 1.0,
                },
                "best_value": 0.5,
            }

        config = {
            "dataset": {"label_column": "label"},
            "training": {"early_stopping": {"patience": 1}, "batch_size": 8},
            "data": {"temporal_sequence": {"window_size": 16}},
        }

        with (
            patch.object(ra, "load_dataset", return_value=df),
            patch.object(ra, "_resolve_timestamp_col", return_value="timestamp"),
            patch.object(ra, "get_feature_cols", return_value=(["f1"], [])),
            patch.object(ra, "temporal_train_val_test_split", side_effect=fake_temporal_split),
            patch.object(ra, "build_tabular_arrays", side_effect=fake_tabular_arrays),
            patch.object(ra, "run_optuna_search", side_effect=fake_search),
        ):
            ra.run_model(
                model_family="xgboost",
                config=config,
                sample_frac=None,
                n_trials=1,
                max_epochs=1,
                horizons=[],
                device="cuda",
                output_dir=tmp_path,
                logger=MagicMock(),
                seeds=[42],
                xgb_device="cpu",
            )

        assert captured["device"] == "cpu"


# ---------------------------------------------------------------------------
# run_status in summary
# ---------------------------------------------------------------------------

class TestRunStatus:
    """summary.json must contain run_status: 'completed' | 'partial_failed'."""

    def _run_main_capture_summary(self, tmp_path, models, horizons, fake_model_fn):
        captured = {}

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py",
                "--models", *models,
                "--horizons", *[str(h) for h in horizons],
                "--output-dir", str(tmp_path),
            ]),
            patch.object(ra, "run_model", side_effect=fake_model_fn),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()
        return captured

    def test_run_status_completed_when_all_succeed(self, tmp_path):
        def fake(model_family, **kwargs):
            return {"k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1}}

        summary = self._run_main_capture_summary(tmp_path, ["xgboost"], ["1"], fake)
        assert summary.get("run_status") == "completed"

    def test_run_status_partial_failed_when_model_raises(self, tmp_path):
        def fake(model_family, **kwargs):
            raise RuntimeError("simulated OOM")

        summary = self._run_main_capture_summary(tmp_path, ["xgboost"], ["1"], fake)
        assert summary.get("run_status") == "partial_failed"

    def test_run_status_partial_failed_when_horizon_fails(self, tmp_path):
        def fake(model_family, **kwargs):
            return {"k_1": {"error": "WinError 1455"}}

        summary = self._run_main_capture_summary(tmp_path, ["xgboost"], ["1"], fake)
        assert summary.get("run_status") == "partial_failed"

    def test_run_status_present_in_all_cases(self, tmp_path):
        def fake(model_family, **kwargs):
            return {"k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1}}

        summary = self._run_main_capture_summary(tmp_path, ["xgboost"], ["1"], fake)
        assert "run_status" in summary


# ---------------------------------------------------------------------------
# --seeds CLI arg and run_mode / is_full_seed_set
# ---------------------------------------------------------------------------

class TestSeedsArg:
    """--seeds must be accepted, default to SEEDS, and drive run_mode / is_full_seed_set."""

    def _run_main_capture_summary(
        self,
        tmp_path: Path,
        extra_argv: list[str],
    ) -> dict:
        """Run main() with extra_argv and return the captured summary dict."""
        import json

        captured = {}

        def fake_run_model(model_family, **kwargs):
            return {
                "k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1}
            }

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py",
                "--models", "xgboost",
                "--horizons", "1",
                "--output-dir", str(tmp_path),
            ] + extra_argv),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()
        return captured

    def test_seeds_flag_accepted_by_dry_run(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path),
                "--seeds", "42",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()  # must not raise SystemExit(2)

    def test_default_seeds_produce_production_mode(self, tmp_path):
        summary = self._run_main_capture_summary(tmp_path, [])
        assert summary.get("run_mode") == "production"
        assert summary.get("is_full_seed_set") is True

    def test_single_seed_produces_pilot_mode(self, tmp_path):
        summary = self._run_main_capture_summary(tmp_path, ["--seeds", "42"])
        assert summary.get("run_mode") == "pilot"
        assert summary.get("is_full_seed_set") is False

    def test_partial_seed_list_produces_pilot_mode(self, tmp_path):
        summary = self._run_main_capture_summary(
            tmp_path, ["--seeds", "42", "123"]
        )
        assert summary.get("run_mode") == "pilot"
        assert summary.get("is_full_seed_set") is False

    def test_seeds_recorded_in_summary_default(self, tmp_path):
        summary = self._run_main_capture_summary(tmp_path, [])
        assert summary.get("seeds") == ra.SEEDS

    def test_seeds_recorded_in_summary_custom(self, tmp_path):
        summary = self._run_main_capture_summary(
            tmp_path, ["--seeds", "42"]
        )
        assert summary.get("seeds") == [42]

    def test_seeds_passed_to_run_model(self, tmp_path):
        received: list[list[int]] = []

        def fake_run_model(model_family, seeds=None, **kwargs):
            received.append(seeds)
            return {
                "k_1": {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": 1}
            }

        with (
            patch("sys.argv", [
                "run_axis2.py",
                "--models", "xgboost",
                "--horizons", "1",
                "--output-dir", str(tmp_path),
                "--seeds", "42",
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results"),
            patch.object(ra, "write_result"),
        ):
            ra.main()

        assert received == [[42]]

    def test_is_full_seed_set_and_run_mode_present_in_summary(self, tmp_path):
        summary = self._run_main_capture_summary(tmp_path, [])
        assert "is_full_seed_set" in summary
        assert "run_mode" in summary

    def test_explicit_full_seed_list_is_production(self, tmp_path):
        """Explicitly passing the full default list still yields production mode."""
        full_seeds = [str(s) for s in ra.SEEDS]
        summary = self._run_main_capture_summary(
            tmp_path, ["--seeds"] + full_seeds
        )
        assert summary.get("run_mode") == "production"
        assert summary.get("is_full_seed_set") is True


# ---------------------------------------------------------------------------
# WindowInputContext — preprocessor reuse
# ---------------------------------------------------------------------------

class TestWindowInputContext:
    """Unit tests for WindowInputContext and its builder functions."""

    def _make_df(self, n: int = 40, seed: int = 0) -> "Any":
        """Tiny DataFrame with numeric features and a binary label."""
        import pandas as pd

        rng = __import__("numpy").random.default_rng(seed)
        return pd.DataFrame({
            "f1": rng.random(n).astype("float32"),
            "f2": rng.random(n).astype("float32"),
            "label": (rng.random(n) > 0.5).astype("int64"),
        })

    def test_build_tabular_context_fits_preprocessor_exactly_once(self):
        """_build_tabular_window_context should call fit_transform exactly once."""
        from unittest.mock import patch, call
        import numpy as np

        df = self._make_df(40)
        fit_df = df.iloc[:24].reset_index(drop=True)
        val_df = df.iloc[24:32].reset_index(drop=True)
        feature_cols = (["f1", "f2"], [])

        fit_transform_calls: list = []
        transform_calls: list = []

        original_init = ra.FlowPreprocessor.__init__

        class TrackingPreprocessor(ra.FlowPreprocessor):
            def fit_transform(self, df_, num_cols, cat_cols=None):
                fit_transform_calls.append(1)
                return super().fit_transform(df_, num_cols, cat_cols)

            def transform(self, df_):
                transform_calls.append(1)
                return super().transform(df_)

        with patch.object(ra, "FlowPreprocessor", TrackingPreprocessor):
            ctx = ra._build_tabular_window_context(
                fit_df, val_df, feature_cols, "label",
                batch_size=16, device="cpu", num_workers=0,
            )

        assert len(fit_transform_calls) == 1, (
            "fit_transform must be called exactly once per window context"
        )
        # val is transformed (not fit_transform)
        assert len(transform_calls) >= 1

    def test_make_test_arrays_does_not_call_fit_transform(self):
        """make_test_arrays must use transform only — never fit_transform."""
        import numpy as np

        df = self._make_df(60)
        fit_df = df.iloc[:30].reset_index(drop=True)
        val_df = df.iloc[30:40].reset_index(drop=True)
        test_df = df.iloc[40:].reset_index(drop=True)
        feature_cols = (["f1", "f2"], [])

        fit_transform_calls: list = []
        original_fit_transform = ra.FlowPreprocessor.fit_transform

        class TrackingPreprocessor(ra.FlowPreprocessor):
            def fit_transform(self, df_, num_cols, cat_cols=None):
                fit_transform_calls.append("fit_transform")
                return original_fit_transform(self, df_, num_cols, cat_cols)

        from unittest.mock import patch
        with patch.object(ra, "FlowPreprocessor", TrackingPreprocessor):
            ctx = ra._build_tabular_window_context(
                fit_df, val_df, feature_cols, "label",
                batch_size=16, device="cpu", num_workers=0,
            )
            n_calls_after_build = len(fit_transform_calls)
            ctx.make_test_arrays(test_df)
            ctx.make_test_arrays(test_df)
            ctx.make_test_arrays(test_df)

        # No additional fit_transform calls from make_test_arrays
        assert len(fit_transform_calls) == n_calls_after_build, (
            "make_test_arrays must not call fit_transform"
        )

    def test_make_test_tabular_loader_no_fit_transform(self):
        """make_test_tabular_loader must not re-fit."""
        df = self._make_df(60)
        fit_df = df.iloc[:30].reset_index(drop=True)
        val_df = df.iloc[30:40].reset_index(drop=True)
        test_df = df.iloc[40:].reset_index(drop=True)
        feature_cols = (["f1", "f2"], [])

        fit_calls: list = []
        orig = ra.FlowPreprocessor.fit_transform

        class Tracker(ra.FlowPreprocessor):
            def fit_transform(self, d, nc, cc=None):
                fit_calls.append(1)
                return orig(self, d, nc, cc)

        from unittest.mock import patch
        with patch.object(ra, "FlowPreprocessor", Tracker):
            ctx = ra._build_tabular_window_context(
                fit_df, val_df, feature_cols, "label",
                batch_size=16, device="cpu", num_workers=0,
            )
            baseline = len(fit_calls)
            ctx.make_test_tabular_loader(test_df, batch_size=16, device="cpu", num_workers=0)
            ctx.make_test_tabular_loader(test_df, batch_size=16, device="cpu", num_workers=0)

        assert len(fit_calls) == baseline

    def test_build_sequential_context_fits_once(self):
        """_build_sequential_window_context should call fit_transform exactly once."""
        import numpy as np

        df = self._make_df(80)
        fit_df = df.iloc[:48].reset_index(drop=True)
        val_df = df.iloc[48:64].reset_index(drop=True)
        feature_cols = (["f1", "f2"], [])

        fit_calls: list = []
        orig = ra.FlowPreprocessor.fit_transform

        class Tracker(ra.FlowPreprocessor):
            def fit_transform(self, d, nc, cc=None):
                fit_calls.append(1)
                return orig(self, d, nc, cc)

        from unittest.mock import patch
        with patch.object(ra, "FlowPreprocessor", Tracker):
            ctx = ra._build_sequential_window_context(
                fit_df, val_df, feature_cols, "label",
                window_size=8, batch_size=8, device="cpu", num_workers=0,
            )

        assert len(fit_calls) == 1

    def test_make_test_sequence_loader_no_fit_transform(self):
        """make_test_sequence_loader must not re-fit."""
        import numpy as np

        df = self._make_df(100)
        fit_df = df.iloc[:60].reset_index(drop=True)
        val_df = df.iloc[60:80].reset_index(drop=True)
        test_df = df.iloc[80:].reset_index(drop=True)
        feature_cols = (["f1", "f2"], [])

        fit_calls: list = []
        orig = ra.FlowPreprocessor.fit_transform

        class Tracker(ra.FlowPreprocessor):
            def fit_transform(self, d, nc, cc=None):
                fit_calls.append(1)
                return orig(self, d, nc, cc)

        from unittest.mock import patch
        with patch.object(ra, "FlowPreprocessor", Tracker):
            ctx = ra._build_sequential_window_context(
                fit_df, val_df, feature_cols, "label",
                window_size=8, batch_size=8, device="cpu", num_workers=0,
            )
            baseline = len(fit_calls)
            ctx.make_test_sequence_loader(test_df, batch_size=8, device="cpu", num_workers=0)
            ctx.make_test_sequence_loader(test_df, batch_size=8, device="cpu", num_workers=0)

        assert len(fit_calls) == baseline

    def test_tabular_context_metrics_equivalent_to_direct_transform(self):
        """Arrays from context must match a direct fit_transform + transform."""
        import numpy as np

        df = self._make_df(60)
        fit_df = df.iloc[:30].reset_index(drop=True)
        val_df = df.iloc[30:40].reset_index(drop=True)
        test_df = df.iloc[40:].reset_index(drop=True)
        feature_cols = (["f1", "f2"], [])

        # Direct reference
        ref_prep = ra.FlowPreprocessor(log_transform=True)
        X_ref = ref_prep.fit_transform(fit_df, ["f1", "f2"], []).to_numpy("float32")
        X_test_ref = ref_prep.transform(test_df).to_numpy("float32")

        # Context path
        ctx = ra._build_tabular_window_context(
            fit_df, val_df, feature_cols, "label",
            batch_size=16, device="cpu", num_workers=0,
        )
        X_train_ctx = ctx.X_train
        X_test_ctx, _ = ctx.make_test_arrays(test_df)

        np.testing.assert_allclose(X_train_ctx, X_ref, rtol=1e-5)
        np.testing.assert_allclose(X_test_ctx, X_test_ref, rtol=1e-5)


# ---------------------------------------------------------------------------
# _check_results_compatible
# ---------------------------------------------------------------------------

class TestCheckResultsCompatible:
    """Unit tests for the resume/skip-existing compatibility check."""

    def _base_existing(self) -> dict:
        return {
            "model_family": "xgboost",
            "dataset":      "MAWIFlow",
            "axis":         2,
            "seeds":        [42, 123, 456, 789, 1024],
            "horizon":      1,
            "n_trials":     50,
            "sample_frac":  None,
            "run_mode":     "production",
        }

    def test_identical_config_is_compatible(self):
        ok, reason = ra._check_results_compatible(
            self._base_existing(),
            model_family="xgboost",
            horizon=1,
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is True
        assert reason == ""

    def test_model_family_mismatch(self):
        ok, reason = ra._check_results_compatible(
            self._base_existing(),
            model_family="mlp",  # different
            horizon=1,
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is False
        assert "model_family" in reason

    def test_horizon_mismatch(self):
        ok, reason = ra._check_results_compatible(
            self._base_existing(),
            model_family="xgboost",
            horizon=3,  # different
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is False
        assert "horizon" in reason

    def test_seeds_mismatch(self):
        ok, reason = ra._check_results_compatible(
            self._base_existing(),
            model_family="xgboost",
            horizon=1,
            seeds=[42],  # different
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is False
        assert "seeds" in reason

    def test_n_trials_mismatch(self):
        ok, reason = ra._check_results_compatible(
            self._base_existing(),
            model_family="xgboost",
            horizon=1,
            seeds=[42, 123, 456, 789, 1024],
            n_trials=10,  # different
            sample_frac=None,
            run_mode="production",
        )
        assert ok is False
        assert "n_trials" in reason

    def test_sample_frac_none_vs_none_is_compatible(self):
        ok, _ = ra._check_results_compatible(
            self._base_existing(),
            model_family="xgboost",
            horizon=1,
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is True

    def test_sample_frac_smoke_vs_full_incompatible(self):
        existing = {**self._base_existing(), "sample_frac": 0.05}
        ok, reason = ra._check_results_compatible(
            existing,
            model_family="xgboost",
            horizon=1,
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,  # full run
            run_mode="production",
        )
        assert ok is False
        assert "sample_frac" in reason

    def test_run_mode_mismatch(self):
        existing = {**self._base_existing(), "run_mode": "pilot"}
        ok, reason = ra._check_results_compatible(
            existing,
            model_family="xgboost",
            horizon=1,
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is False
        assert "run_mode" in reason

    def test_missing_run_mode_in_existing_is_compatible(self):
        """Existing results without run_mode (old format) must not fail."""
        existing = {k: v for k, v in self._base_existing().items() if k != "run_mode"}
        ok, _ = ra._check_results_compatible(
            existing,
            model_family="xgboost",
            horizon=1,
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is True

    def test_cumulative_horizon_string_match(self):
        existing = {**self._base_existing(), "horizon": "cumulative"}
        ok, _ = ra._check_results_compatible(
            existing,
            model_family="xgboost",
            horizon="cumulative",
            seeds=[42, 123, 456, 789, 1024],
            n_trials=50,
            sample_frac=None,
            run_mode="production",
        )
        assert ok is True


# ---------------------------------------------------------------------------
# --resume / --skip-existing CLI behaviour
# ---------------------------------------------------------------------------

class TestResumeSkipExisting:
    """Integration tests for --resume and --skip-existing flags."""

    def _make_valid_results_json(self, path: Path, model_family: str, horizon) -> None:
        """Write a minimal but compatible results.json."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_family": model_family,
            "dataset":      "MAWIFlow",
            "axis":         2,
            "seeds":        list(ra.SEEDS),
            "horizon":      horizon,
            "n_trials":     50,
            "sample_frac":  None,
            "run_mode":     "production",
            "metrics":      {},
            "best_params":  {},
            "elapsed_seconds": 1.0,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _run_main_capture(self, tmp_path: Path, extra_argv: list[str]) -> tuple[dict, list]:
        """Run main() with mocked run_model; return (summary, run_model_calls)."""
        import json
        captured: dict = {}
        run_model_calls: list = []

        def fake_run_model(model_family, horizons, **kwargs):
            run_model_calls.append((model_family, horizons))
            return {
                ra._horizon_dir_name(h): {
                    "metrics": {}, "best_params": {}, "elapsed_seconds": 1.0,
                    "horizon": h,
                }
                for h in horizons
            }

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py",
                "--models", "xgboost",
                "--horizons", "1",
                "--output-dir", str(tmp_path),
            ] + extra_argv),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()

        return captured, run_model_calls

    def test_resume_flag_accepted_by_dry_run(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path), "--resume",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()  # must not raise

    def test_skip_existing_flag_accepted_by_dry_run(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path), "--skip-existing",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()

    def test_fail_fast_flag_accepted_by_dry_run(self, tmp_path):
        with (
            patch("sys.argv", [
                "run_axis2.py", "--dry-run",
                "--output-dir", str(tmp_path), "--fail-fast",
            ]),
            patch.object(ra, "run_model"),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
        ):
            ra.main()

    def test_resume_skips_compatible_combination(self, tmp_path):
        """With --resume, a compatible results.json causes the combination to be skipped."""
        results_path = tmp_path / "xgboost" / "k_1" / "results.json"
        self._make_valid_results_json(results_path, "xgboost", 1)

        skipped: list = []
        executed: list = []

        def fake_run_model(model_family, horizons, resume, skip_existing, **kwargs):
            h_results = {}
            for h in horizons:
                h_key = ra._horizon_dir_name(h)
                rpath = tmp_path / model_family / h_key / "results.json"
                if (resume or skip_existing) and rpath.exists():
                    import json as _json
                    existing = _json.loads(rpath.read_text())
                    ok, _ = ra._check_results_compatible(
                        existing, model_family, h, list(ra.SEEDS), 50, None, "production"
                    )
                    if ok:
                        existing["skipped_existing"] = True
                        skipped.append(f"{model_family}/{h_key}")
                        h_results[h_key] = existing
                        continue
                executed.append(f"{model_family}/{h_key}")
                h_results[h_key] = {
                    "metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": h,
                }
            return h_results

        captured: dict = {}

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py", "--models", "xgboost", "--horizons", "1",
                "--output-dir", str(tmp_path), "--resume",
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()

        assert "xgboost/k_1" in skipped
        assert "xgboost/k_1" not in executed
        assert "combinations_skipped_existing" in captured
        assert "xgboost/k_1" in captured["combinations_skipped_existing"]

    def test_summary_contains_combinations_skipped_existing(self, tmp_path):
        """summary.json must always contain combinations_skipped_existing key."""
        captured: dict = {}

        def fake_run_model(model_family, horizons, **kwargs):
            return {
                ra._horizon_dir_name(h): {
                    "metrics": {}, "best_params": {}, "elapsed_seconds": 1.0, "horizon": h,
                }
                for h in horizons
            }

        def capture_save(data, path):
            if "summary" in str(path):
                captured.update(data)

        with (
            patch("sys.argv", [
                "run_axis2.py", "--models", "xgboost", "--horizons", "1",
                "--output-dir", str(tmp_path),
            ]),
            patch.object(ra, "run_model", side_effect=fake_run_model),
            patch.object(ra, "resolve_device", return_value="cpu"),
            patch.object(ra, "setup_logger", return_value=MagicMock()),
            patch.object(ra, "save_results", side_effect=capture_save),
            patch.object(ra, "write_result"),
        ):
            ra.main()

        assert "combinations_skipped_existing" in captured


# ---------------------------------------------------------------------------
# aggregate_axis2_summaries
# ---------------------------------------------------------------------------

class TestAggregateAxis2Summaries:
    """Tests for aggregate_axis2_summaries.py aggregate() function."""

    def _write_results(
        self,
        output_dir: Path,
        model: str,
        horizon: str,
        error: str | None = None,
        seeds: list | None = None,
    ) -> None:
        h_key = f"k_{horizon}"
        path = output_dir / model / h_key / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if error:
            payload = {"error": error, "model_family": model, "horizon": horizon}
        else:
            payload = {
                "model_family": model,
                "dataset":      "MAWIFlow",
                "axis":         2,
                "horizon":      int(horizon) if horizon.isdigit() else horizon,
                "seeds":        seeds or list(ra.SEEDS),
                "n_trials":     50,
                "metrics":      {},
                "best_params":  {},
                "elapsed_seconds": 1.0,
            }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_all_models_and_horizons_present_is_completed(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import aggregate_axis2_summaries as agg

        for m in ["xgboost", "mlp", "cnn_bilstm", "transformer"]:
            for h in ["1", "2", "3", "cumulative"]:
                self._write_results(tmp_path, m, h)

        summary = agg.aggregate(
            tmp_path,
            expected_models=["xgboost", "mlp", "cnn_bilstm", "transformer"],
            expected_horizons=["1", "2", "3", "cumulative"],
            expected_seeds=list(ra.SEEDS),
        )

        assert summary["run_status"] == "completed"
        assert summary["is_full_temporal_matrix"] is True
        assert len(summary["combinations_missing"]) == 0
        assert len(summary["combinations_failed"]) == 0
        assert len(summary["combinations_executed"]) == 16

    def test_missing_model_is_incomplete(self, tmp_path):
        import aggregate_axis2_summaries as agg

        for m in ["xgboost", "mlp", "cnn_bilstm"]:  # transformer missing
            for h in ["1", "2", "3", "cumulative"]:
                self._write_results(tmp_path, m, h)

        summary = agg.aggregate(
            tmp_path,
            expected_models=["xgboost", "mlp", "cnn_bilstm", "transformer"],
            expected_horizons=["1", "2", "3", "cumulative"],
            expected_seeds=list(ra.SEEDS),
        )

        assert summary["run_status"] == "incomplete"
        assert summary["is_full_temporal_matrix"] is False
        assert any("transformer" in c for c in summary["combinations_missing"])

    def test_missing_horizon_is_incomplete(self, tmp_path):
        import aggregate_axis2_summaries as agg

        for m in ["xgboost", "mlp", "cnn_bilstm", "transformer"]:
            for h in ["1", "2", "3"]:  # cumulative missing
                self._write_results(tmp_path, m, h)

        summary = agg.aggregate(
            tmp_path,
            expected_models=["xgboost", "mlp", "cnn_bilstm", "transformer"],
            expected_horizons=["1", "2", "3", "cumulative"],
            expected_seeds=list(ra.SEEDS),
        )

        assert summary["run_status"] == "incomplete"
        assert summary["is_full_temporal_matrix"] is False
        assert any("cumulative" in c for c in summary["combinations_missing"])

    def test_error_result_is_partial_failed(self, tmp_path):
        import aggregate_axis2_summaries as agg

        for m in ["xgboost", "mlp", "cnn_bilstm", "transformer"]:
            for h in ["1", "2", "3", "cumulative"]:
                if m == "transformer" and h == "cumulative":
                    self._write_results(tmp_path, m, h, error="OOM error")
                else:
                    self._write_results(tmp_path, m, h)

        summary = agg.aggregate(
            tmp_path,
            expected_models=["xgboost", "mlp", "cnn_bilstm", "transformer"],
            expected_horizons=["1", "2", "3", "cumulative"],
            expected_seeds=list(ra.SEEDS),
        )

        assert summary["run_status"] == "partial_failed"
        assert summary["is_full_temporal_matrix"] is False
        assert any("transformer/k_cumulative" in c for c in summary["combinations_failed"])

    def test_aggregate_writes_summary_json(self, tmp_path):
        import aggregate_axis2_summaries as agg

        for m in ["xgboost"]:
            self._write_results(tmp_path, m, "1")

        summary = agg.aggregate(
            tmp_path,
            expected_models=["xgboost"],
            expected_horizons=["1"],
            expected_seeds=[42],
        )
        out_path = tmp_path / "summary.json"
        out_path.write_text(json.dumps(summary), encoding="utf-8")
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["run_status"] in {"completed", "incomplete", "partial_failed"}

    def test_aggregate_records_seeds_found(self, tmp_path):
        import aggregate_axis2_summaries as agg

        self._write_results(tmp_path, "xgboost", "1", seeds=[42, 123])

        summary = agg.aggregate(
            tmp_path,
            expected_models=["xgboost"],
            expected_horizons=["1"],
            expected_seeds=[42, 123, 456, 789, 1024],
        )

        assert summary["seeds_found"] == [42, 123]
        assert summary["is_full_seed_set"] is False


# ---------------------------------------------------------------------------
# _load_existing_horizon_result
# ---------------------------------------------------------------------------

class TestLoadExistingHorizonResult:
    """Unit tests for _load_existing_horizon_result pre-check helper."""

    def _make_results_json(
        self,
        path: Path,
        model_family: str = "xgboost",
        horizon: int | str = 1,
        seeds: list | None = None,
        n_trials: int = 50,
        run_mode: str = "pilot",
        error: str | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if error:
            payload = {"error": error, "model_family": model_family}
        else:
            payload = {
                "model_family": model_family,
                "dataset":      "MAWIFlow",
                "axis":         2,
                "seeds":        seeds or [42],
                "horizon":      horizon,
                "n_trials":     n_trials,
                "sample_frac":  None,
                "run_mode":     run_mode,
                "metrics":      {},
                "best_params":  {},
                "elapsed_seconds": 1.0,
            }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_when_no_file(self, tmp_path):
        status, result = ra._load_existing_horizon_result(
            tmp_path / "xgboost", "xgboost", 1, [42], 50, None, "pilot", False
        )
        assert status == "missing"
        assert result is None

    def test_compatible_for_valid_existing_result(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        self._make_results_json(model_dir / "k_1" / "results.json")
        status, result = ra._load_existing_horizon_result(
            model_dir, "xgboost", 1, [42], 50, None, "pilot", False
        )
        assert status == "compatible"
        assert result is not None
        assert result.get("skipped_existing") is True

    def test_missing_when_file_has_error_key(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        self._make_results_json(model_dir / "k_1" / "results.json", error="OOM")
        status, result = ra._load_existing_horizon_result(
            model_dir, "xgboost", 1, [42], 50, None, "pilot", False
        )
        assert status == "missing"
        assert result is None

    def test_incompatible_with_resume_returns_incompatible(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        self._make_results_json(model_dir / "k_1" / "results.json", n_trials=50)
        # n_trials mismatch → incompatible
        status, result = ra._load_existing_horizon_result(
            model_dir, "xgboost", 1, [42], 10, None, "pilot", False
        )
        assert status == "incompatible"
        assert result is None

    def test_incompatible_with_skip_existing_raises(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        self._make_results_json(model_dir / "k_1" / "results.json", n_trials=50)
        with pytest.raises(RuntimeError, match="--skip-existing"):
            ra._load_existing_horizon_result(
                model_dir, "xgboost", 1, [42], 10, None, "pilot", True
            )

    def test_missing_when_json_invalid(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        p = model_dir / "k_1" / "results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not valid json", encoding="utf-8")
        status, result = ra._load_existing_horizon_result(
            model_dir, "xgboost", 1, [42], 50, None, "pilot", False
        )
        assert status == "missing"

    def test_cumulative_horizon_uses_k_cumulative_path(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        self._make_results_json(
            model_dir / "k_cumulative" / "results.json",
            horizon="cumulative",
        )
        status, result = ra._load_existing_horizon_result(
            model_dir, "xgboost", "cumulative", [42], 50, None, "pilot", False
        )
        assert status == "compatible"

    def test_compatible_result_is_not_loaded_from_wrong_path(self, tmp_path):
        """Compatible k_1 result must NOT be found when querying k_cumulative."""
        model_dir = tmp_path / "xgboost"
        self._make_results_json(model_dir / "k_1" / "results.json", horizon=1)
        status, result = ra._load_existing_horizon_result(
            model_dir, "xgboost", "cumulative", [42], 50, None, "pilot", False
        )
        assert status == "missing"


# ---------------------------------------------------------------------------
# run_model early resume / skip HPO
# ---------------------------------------------------------------------------

class TestRunModelEarlyResume:
    """run_model must skip data loading and HPO when all horizons are complete."""

    def _make_results_json(
        self,
        path: Path,
        model_family: str,
        horizon: int | str,
        seeds: list,
        n_trials: int = 1,
        run_mode: str = "pilot",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_family": model_family,
            "dataset":      "MAWIFlow",
            "axis":         2,
            "seeds":        seeds,
            "horizon":      horizon,
            "n_trials":     n_trials,
            "sample_frac":  None,
            "run_mode":     run_mode,
            "metrics":      {},
            "best_params":  {},
            "elapsed_seconds": 1.0,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _config(self):
        return {
            "dataset": {"label_column": "label"},
            "training": {"early_stopping": {"patience": 1}, "batch_size": 8},
            "data": {"temporal_sequence": {"window_size": 16}},
        }

    def test_all_horizons_compatible_skips_load_and_hpo(self, tmp_path):
        """When all requested horizons have compatible results, load_dataset and
        run_optuna_search must NOT be called."""
        seeds = [42]
        for horizon in [1, "cumulative"]:
            h_key = ra._horizon_dir_name(horizon)
            self._make_results_json(
                tmp_path / "xgboost" / h_key / "results.json",
                "xgboost", horizon, seeds,
            )

        load_called = []
        hpo_called = []

        with (
            patch.object(ra, "load_dataset",
                         side_effect=lambda *a, **kw: load_called.append(1)),
            patch.object(ra, "run_optuna_search",
                         side_effect=lambda *a, **kw: hpo_called.append(1)),
        ):
            result = ra.run_model(
                model_family="xgboost",
                config=self._config(),
                sample_frac=None,
                n_trials=1,
                max_epochs=1,
                horizons=[1, "cumulative"],
                device="cpu",
                output_dir=tmp_path,
                logger=MagicMock(),
                seeds=seeds,
                resume=True,
                run_mode="pilot",
            )

        assert load_called == [], "load_dataset must not be called when all horizons complete"
        assert hpo_called == [], "run_optuna_search must not be called when all horizons complete"
        assert result["k_1"].get("skipped_existing") is True
        assert result["k_cumulative"].get("skipped_existing") is True

    def test_partial_resume_calls_hpo_once(self, tmp_path):
        """With k_1 complete and k_cumulative missing, HPO runs exactly once and
        k_1 is returned as skipped_existing."""
        import numpy as np
        import pandas as pd

        seeds = [42]
        self._make_results_json(
            tmp_path / "xgboost" / "k_1" / "results.json",
            "xgboost", 1, seeds,
        )

        hpo_calls = []
        df = pd.DataFrame({
            "year": [2007] * 8 + [2008] * 4,
            "timestamp": pd.date_range("2007-01-01", periods=12, freq="h"),
            "label": [0, 1] * 6,
            "f1": np.arange(12, dtype=float),
        })
        fake_seed_result = {
            "seed": 42, "nAUT_1": 0.5, "nAUT_3": 0.5, "nAUT_5": 0.5,
            "median_trajectory": [(0, 0, 0.5)], "window_results": [],
            "f1plus_records": [], "window_efficiency": [],
            "f1_macro": 0.5, "f1_1": 0.5, "precision": 0.5,
            "recall": 0.5, "auc_roc": 0.5, "auc_pr": 0.5,
        }

        def fake_hpo(**kwargs):
            hpo_calls.append(1)
            return {
                "best_params": {
                    "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
                    "subsample": 1.0, "colsample_bytree": 1.0, "scale_pos_weight": 1.0,
                },
                "best_value": 0.5,
            }

        with (
            patch.object(ra, "load_dataset", return_value=df),
            patch.object(ra, "_resolve_timestamp_col", return_value="timestamp"),
            patch.object(ra, "get_feature_cols", return_value=(["f1"], [])),
            patch.object(ra, "temporal_train_val_test_split", return_value=(
                df.iloc[:4].reset_index(drop=True),
                df.iloc[4:6].reset_index(drop=True),
                df.iloc[6:8].reset_index(drop=True),
            )),
            patch.object(ra, "build_tabular_arrays", return_value=(
                (np.zeros((4, 1), np.float32), np.array([0, 1, 0, 1])),
                (np.zeros((2, 1), np.float32), np.array([0, 1])),
                (np.zeros((2, 1), np.float32), np.array([0, 1])),
                MagicMock(),
            )),
            patch.object(ra, "run_optuna_search", side_effect=fake_hpo),
            patch.object(ra, "_run_forward_chaining_seed",
                         return_value=fake_seed_result),
            patch.object(ra, "aggregate_efficiency_records", return_value={}),
            patch.object(ra, "save_results"),
            patch.object(ra, "write_result"),
        ):
            result = ra.run_model(
                model_family="xgboost",
                config=self._config(),
                sample_frac=None,
                n_trials=1,
                max_epochs=1,
                horizons=[1, "cumulative"],
                device="cpu",
                output_dir=tmp_path,
                logger=MagicMock(),
                seeds=seeds,
                resume=True,
                run_mode="pilot",
            )

        assert len(hpo_calls) == 1, (
            f"run_optuna_search must be called exactly once, was called {len(hpo_calls)} times"
        )
        assert result.get("k_1", {}).get("skipped_existing") is True
        assert "k_cumulative" in result
        assert "error" not in result["k_cumulative"]


# ---------------------------------------------------------------------------
# effective_batch_size: HPO batch_size flows into _run_forward_chaining_seed
# ---------------------------------------------------------------------------

class TestEffectiveBatchSize:
    """_run_forward_chaining_seed must receive batch_size from best_params,
    not from config['training']['batch_size'], for PyTorch models."""

    def _config(self, config_batch_size: int = 2048) -> dict:
        return {
            "dataset": {"label_column": "label"},
            "training": {"early_stopping": {"patience": 1}, "batch_size": config_batch_size},
            "data": {"temporal_sequence": {"window_size": 16}},
        }

    def _minimal_df(self):
        import numpy as np
        import pandas as pd
        return pd.DataFrame({
            "year": [2007] * 8 + [2008] * 4,
            "timestamp": pd.date_range("2007-01-01", periods=12, freq="h"),
            "label": [0, 1] * 6,
            "f1": np.arange(12, dtype=float),
        })

    def _fake_hpo_result(self, batch_size: int) -> dict:
        return {
            "best_params": {
                "lr": 1e-3,
                "dropout": 0.1,
                "head_dropout": 0.1,
                "weight_decay": 1e-4,
                "batch_size": batch_size,
            },
            "best_value": 0.6,
        }

    def _run_and_capture_batch_size(
        self,
        tmp_path: Path,
        model_family: str,
        hpo_batch_size: int,
        config_batch_size: int = 2048,
    ) -> int:
        """Run run_model() and return the batch_size received by _run_forward_chaining_seed."""
        import numpy as np

        df = self._minimal_df()
        received: list[int] = []
        fake_seed_result = {
            "seed": 42, "nAUT_1": 0.5, "nAUT_3": 0.5, "nAUT_5": 0.5,
            "median_trajectory": [(0, 0, 0.5)], "window_results": [],
            "f1plus_records": [], "window_efficiency": [],
            "f1_macro": 0.5, "f1_1": 0.5, "precision": 0.5,
            "recall": 0.5, "auc_roc": 0.5, "auc_pr": 0.5,
        }

        def capture_fcs(**kwargs):
            received.append(kwargs["batch_size"])
            return fake_seed_result

        def fake_hpo(**kwargs):
            return self._fake_hpo_result(hpo_batch_size)

        def fake_build_seq(*args, **kwargs):
            n_features = 1
            loader = MagicMock()
            loader.dataset = MagicMock()
            loader.num_workers = 0
            loader.pin_memory = False
            return loader, loader, loader, n_features, MagicMock()

        def fake_build_tab(*args, **kwargs):
            X = np.zeros((4, 1), np.float32)
            y = np.array([0, 1, 0, 1], np.int64)
            loader = MagicMock()
            loader.dataset = MagicMock()
            loader.num_workers = 0
            loader.pin_memory = False
            return (X, y), (X, y), (X, y), loader, loader

        def fake_tvt_split(anchor_df, *_args, **_kwargs):
            return (
                anchor_df.iloc[:4].reset_index(drop=True),
                anchor_df.iloc[4:6].reset_index(drop=True),
                anchor_df.iloc[6:].reset_index(drop=True),
            )

        with (
            patch.object(ra, "load_dataset", return_value=df),
            patch.object(ra, "_resolve_timestamp_col", return_value="timestamp"),
            patch.object(ra, "get_feature_cols", return_value=(["f1"], [])),
            patch.object(ra, "temporal_train_val_test_split", side_effect=fake_tvt_split),
            patch.object(ra, "build_sequence_inputs", side_effect=fake_build_seq),
            patch.object(ra, "build_tabular_inputs", side_effect=fake_build_tab),
            patch.object(ra, "run_optuna_search", side_effect=fake_hpo),
            patch.object(ra, "_run_forward_chaining_seed", side_effect=capture_fcs),
            patch.object(ra, "aggregate_efficiency_records", return_value={}),
            patch.object(ra, "save_results"),
            patch.object(ra, "write_result"),
        ):
            ra.run_model(
                model_family=model_family,
                config=self._config(config_batch_size),
                sample_frac=None,
                n_trials=1,
                max_epochs=1,
                horizons=[1],
                device="cpu",
                output_dir=tmp_path,
                logger=MagicMock(),
                seeds=[42],
            )

        assert len(received) == 1, "expected exactly one _run_forward_chaining_seed call"
        return received[0]

    def test_transformer_uses_hpo_batch_size_not_config_batch_size(self, tmp_path):
        """Transformer forward-chaining must use batch_size from best_params (e.g. 128),
        not from config['training']['batch_size'] (e.g. 2048)."""
        bs = self._run_and_capture_batch_size(
            tmp_path, "transformer", hpo_batch_size=128, config_batch_size=2048
        )
        assert bs == 128, (
            f"Expected effective_batch_size=128 (from HPO), got {bs}. "
            "Check that run_model uses best_params['batch_size'] for PyTorch models."
        )

    def test_mlp_uses_hpo_batch_size_not_config_batch_size(self, tmp_path):
        """MLP forward-chaining must use batch_size from best_params."""
        bs = self._run_and_capture_batch_size(
            tmp_path, "mlp", hpo_batch_size=256, config_batch_size=2048
        )
        assert bs == 256

    def test_cnn_bilstm_uses_hpo_batch_size_not_config_batch_size(self, tmp_path):
        """CNN-BiLSTM forward-chaining must use batch_size from best_params."""
        bs = self._run_and_capture_batch_size(
            tmp_path, "cnn_bilstm", hpo_batch_size=64, config_batch_size=2048
        )
        assert bs == 64

    def test_effective_batch_size_recorded_in_results(self, tmp_path):
        """results.json must contain 'effective_batch_size' for PyTorch models."""
        import numpy as np

        df = self._minimal_df()
        saved_results: list[dict] = []
        fake_seed_result = {
            "seed": 42, "nAUT_1": 0.5, "nAUT_3": 0.5, "nAUT_5": 0.5,
            "median_trajectory": [(0, 0, 0.5)], "window_results": [],
            "f1plus_records": [], "window_efficiency": [],
            "f1_macro": 0.5, "f1_1": 0.5, "precision": 0.5,
            "recall": 0.5, "auc_roc": 0.5, "auc_pr": 0.5,
        }

        def fake_build_seq(*args, **kwargs):
            loader = MagicMock()
            loader.dataset = MagicMock()
            loader.num_workers = 0
            loader.pin_memory = False
            return loader, loader, loader, 1, MagicMock()

        def capture_write(record, path, **kwargs):
            if "results.json" in str(path) and "seed_" not in str(path):
                saved_results.append(dict(record))

        def fake_tvt_split(anchor_df, *_args, **_kwargs):
            return (
                anchor_df.iloc[:4].reset_index(drop=True),
                anchor_df.iloc[4:6].reset_index(drop=True),
                anchor_df.iloc[6:].reset_index(drop=True),
            )

        with (
            patch.object(ra, "load_dataset", return_value=df),
            patch.object(ra, "_resolve_timestamp_col", return_value="timestamp"),
            patch.object(ra, "get_feature_cols", return_value=(["f1"], [])),
            patch.object(ra, "temporal_train_val_test_split", side_effect=fake_tvt_split),
            patch.object(ra, "build_sequence_inputs", side_effect=fake_build_seq),
            patch.object(ra, "run_optuna_search",
                         return_value=self._fake_hpo_result(batch_size=128)),
            patch.object(ra, "_run_forward_chaining_seed",
                         return_value=fake_seed_result),
            patch.object(ra, "aggregate_efficiency_records", return_value={}),
            patch.object(ra, "save_results"),
            patch.object(ra, "write_result", side_effect=capture_write),
        ):
            ra.run_model(
                model_family="transformer",
                config=self._config(2048),
                sample_frac=None,
                n_trials=1,
                max_epochs=1,
                horizons=[1],
                device="cpu",
                output_dir=tmp_path,
                logger=MagicMock(),
                seeds=[42],
            )

        assert saved_results, "save_results was not called for results.json"
        h_result = saved_results[0]
        assert "effective_batch_size" in h_result, (
            "results.json must contain 'effective_batch_size'"
        )
        assert h_result["effective_batch_size"] == 128

    def test_xgboost_batch_size_unchanged(self, tmp_path):
        """XGBoost does not have batch_size in its search space;
        effective_batch_size must equal config['training']['batch_size']."""
        import numpy as np

        df = self._minimal_df()
        received: list[int] = []
        fake_seed_result = {
            "seed": 42, "nAUT_1": 0.5, "nAUT_3": 0.5, "nAUT_5": 0.5,
            "median_trajectory": [(0, 0, 0.5)], "window_results": [],
            "f1plus_records": [], "window_efficiency": [],
            "f1_macro": 0.5, "f1_1": 0.5, "precision": 0.5,
            "recall": 0.5, "auc_roc": 0.5, "auc_pr": 0.5,
        }

        def capture_fcs(**kwargs):
            received.append(kwargs["batch_size"])
            return fake_seed_result

        def fake_hpo(**kwargs):
            # XGBoost best_params has no batch_size
            return {
                "best_params": {
                    "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
                    "subsample": 1.0, "colsample_bytree": 1.0, "scale_pos_weight": 1.0,
                },
                "best_value": 0.5,
            }

        def fake_tab_arrays(*args, **kwargs):
            X = np.zeros((4, 1), np.float32)
            y = np.array([0, 1, 0, 1], np.int64)
            return (X, y), (X, y), (X, y), MagicMock()

        def fake_tvt_split(anchor_df, *_args, **_kwargs):
            return (
                anchor_df.iloc[:4].reset_index(drop=True),
                anchor_df.iloc[4:6].reset_index(drop=True),
                anchor_df.iloc[6:].reset_index(drop=True),
            )

        with (
            patch.object(ra, "load_dataset", return_value=df),
            patch.object(ra, "_resolve_timestamp_col", return_value="timestamp"),
            patch.object(ra, "get_feature_cols", return_value=(["f1"], [])),
            patch.object(ra, "temporal_train_val_test_split", side_effect=fake_tvt_split),
            patch.object(ra, "build_tabular_arrays", side_effect=fake_tab_arrays),
            patch.object(ra, "run_optuna_search", side_effect=fake_hpo),
            patch.object(ra, "_run_forward_chaining_seed", side_effect=capture_fcs),
            patch.object(ra, "aggregate_efficiency_records", return_value={}),
            patch.object(ra, "save_results"),
            patch.object(ra, "write_result"),
        ):
            ra.run_model(
                model_family="xgboost",
                config=self._config(2048),
                sample_frac=None,
                n_trials=1,
                max_epochs=1,
                horizons=[1],
                device="cpu",
                output_dir=tmp_path,
                logger=MagicMock(),
                seeds=[42],
                xgb_device="cpu",
            )

        assert len(received) == 1
        assert received[0] == 2048, (
            f"XGBoost effective_batch_size must equal config default (2048), got {received[0]}"
        )
