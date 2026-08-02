"""Tests for GPU diagnostic helpers in the experiment CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_run_experiment_module():
    script_path = Path(__file__).parents[1] / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_experiment = _load_run_experiment_module()


def test_resolve_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(run_experiment.torch.cuda, "is_available", lambda: False)

    device = run_experiment.resolve_device({"training": {"device": "auto"}})

    assert device == "cpu"


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(run_experiment.torch.cuda, "is_available", lambda: True)

    device = run_experiment.resolve_device({"training": {"device": "auto"}})

    assert device == "cuda"


def test_resolve_device_cuda_override_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr(run_experiment.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        run_experiment.resolve_device({}, device_override="cuda")


def test_torch_cuda_tensor_ops_reports_unavailable(monkeypatch):
    monkeypatch.setattr(run_experiment.torch.cuda, "is_available", lambda: False)

    result = run_experiment.test_torch_cuda_tensor_ops()

    assert not result["ok"]
    assert "is_available" in result["error"]


def test_gpu_diagnostics_can_skip_xgboost(monkeypatch):
    monkeypatch.setattr(
        run_experiment,
        "collect_torch_cuda_info",
        lambda: {"cuda_available": False},
    )
    monkeypatch.setattr(
        run_experiment,
        "test_torch_cuda_tensor_ops",
        lambda device: {"name": "torch_cuda_tensor_ops", "ok": True, "device": device},
    )
    monkeypatch.setattr(
        run_experiment,
        "test_cuda_dataloader_transfer",
        lambda device: {"name": "cuda_dataloader_transfer", "ok": True, "device": device},
    )

    report = run_experiment.run_gpu_diagnostics(device="cuda", include_xgboost=False)

    assert report["ok"]
    assert len(report["checks"]) == 2
    assert {check["name"] for check in report["checks"]} == {
        "torch_cuda_tensor_ops",
        "cuda_dataloader_transfer",
    }
