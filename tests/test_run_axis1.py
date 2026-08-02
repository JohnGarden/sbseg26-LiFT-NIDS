"""Tests for scripts/run_axis1.py CLI plumbing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_axis1 as ra


def test_num_workers_cli_passed_to_run_model(tmp_path):
    captured_kwargs = {}

    def fake_run_model(**kwargs):
        captured_kwargs.update(kwargs)
        return {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0}

    with (
        patch(
            "sys.argv",
            [
                "run_axis1.py",
                "--models",
                "cnn_bilstm",
                "--output-dir",
                str(tmp_path),
                "--num-workers",
                "0",
            ],
        ),
        patch.object(ra, "run_model", side_effect=fake_run_model),
        patch.object(ra, "resolve_device", return_value="cpu"),
        patch.object(ra, "setup_logger", return_value=MagicMock()),
        patch.object(ra, "save_results"),
    ):
        ra.main()

    assert captured_kwargs["num_workers"] == 0


def test_num_workers_recorded_in_summary(tmp_path):
    captured_summary = {}

    def fake_run_model(**kwargs):
        return {"metrics": {}, "best_params": {}, "elapsed_seconds": 1.0}

    def capture_save(data, path):
        if "summary" in str(path):
            captured_summary.update(data)

    with (
        patch(
            "sys.argv",
            [
                "run_axis1.py",
                "--models",
                "cnn_bilstm",
                "--output-dir",
                str(tmp_path),
                "--num-workers",
                "0",
            ],
        ),
        patch.object(ra, "run_model", side_effect=fake_run_model),
        patch.object(ra, "resolve_device", return_value="cpu"),
        patch.object(ra, "setup_logger", return_value=MagicMock()),
        patch.object(ra, "save_results", side_effect=capture_save),
    ):
        ra.main()

    assert captured_summary["num_workers"] == 0
