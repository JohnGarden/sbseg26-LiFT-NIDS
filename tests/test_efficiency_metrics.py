"""Tests for efficiency metrics: build_efficiency_record, aggregate_efficiency_records,
count_trainable_params, measure_pytorch_inference, and XGBoostWrapper efficiency attributes."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lift_nids.evaluation.efficiency import (
    aggregate_efficiency_records,
    build_efficiency_record,
    count_trainable_params,
    get_peak_gpu_memory_bytes,
    measure_pytorch_inference,
    reset_peak_gpu_memory,
    sync_cuda_if_needed,
)
from lift_nids.models.xgboost_wrapper import XGBoostWrapper


# ---------------------------------------------------------------------------
# count_trainable_params
# ---------------------------------------------------------------------------

class TestCountTrainableParams:
    def test_linear_layer(self):
        m = nn.Linear(10, 5)  # 10*5 + 5 = 55
        assert count_trainable_params(m) == 55

    def test_frozen_params_excluded(self):
        m = nn.Linear(10, 5)
        for p in m.parameters():
            p.requires_grad = False
        assert count_trainable_params(m) == 0

    def test_mixed_frozen_unfrozen(self):
        m = nn.Linear(10, 5)
        m.bias.requires_grad = False
        # only weight: 10*5 = 50
        assert count_trainable_params(m) == 50

    def test_sequential(self):
        m = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        expected = (4 * 8 + 8) + (8 * 2 + 2)  # 40 + 18 = 58
        assert count_trainable_params(m) == expected

    def test_empty_module(self):
        m = nn.Sequential()
        assert count_trainable_params(m) == 0


# ---------------------------------------------------------------------------
# build_efficiency_record
# ---------------------------------------------------------------------------

class TestBuildEfficiencyRecord:
    def _make(self, **kwargs):
        defaults = dict(
            n_params=1000,
            train_time_s=5.0,
            infer_time_s=0.01,
            n_train_samples=500,
            n_infer_samples=100,
            peak_gpu_mem_bytes=0,
            device="cpu",
        )
        defaults.update(kwargs)
        return build_efficiency_record(**defaults)

    def test_has_all_required_fields(self):
        rec = self._make()
        required = {
            "n_params", "train_time_s", "infer_time_s", "infer_time_s_per_sample",
            "peak_gpu_mem_bytes", "device", "n_train_samples", "n_infer_samples",
        }
        assert required.issubset(rec.keys())

    def test_infer_time_per_sample_correct(self):
        rec = self._make(infer_time_s=0.1, n_infer_samples=50)
        assert abs(rec["infer_time_s_per_sample"] - 0.002) < 1e-9

    def test_infer_time_per_sample_none_when_zero_samples(self):
        rec = self._make(n_infer_samples=0)
        assert rec["infer_time_s_per_sample"] is None

    def test_train_time_rounded_to_4dp(self):
        rec = self._make(train_time_s=1.23456789)
        assert rec["train_time_s"] == round(1.23456789, 4)

    def test_infer_time_rounded_to_6dp(self):
        rec = self._make(infer_time_s=0.00123456789)
        assert rec["infer_time_s"] == round(0.00123456789, 6)

    def test_peak_gpu_mem_is_int(self):
        rec = self._make(peak_gpu_mem_bytes=12345678)
        assert isinstance(rec["peak_gpu_mem_bytes"], int)
        assert rec["peak_gpu_mem_bytes"] == 12345678

    def test_device_is_str(self):
        rec = self._make(device=torch.device("cpu"))
        assert isinstance(rec["device"], str)
        assert rec["device"] == "cpu"

    def test_n_train_samples_is_int(self):
        rec = self._make(n_train_samples=np.int64(500))
        assert isinstance(rec["n_train_samples"], int)

    def test_n_infer_samples_is_int(self):
        rec = self._make(n_infer_samples=np.int64(100))
        assert isinstance(rec["n_infer_samples"], int)


# ---------------------------------------------------------------------------
# aggregate_efficiency_records
# ---------------------------------------------------------------------------

class TestAggregateEfficiencyRecords:
    def _rec(self, train_time_s, infer_time_s, n_infer_samples=100, peak_gpu=0):
        return build_efficiency_record(
            n_params=256,
            train_time_s=train_time_s,
            infer_time_s=infer_time_s,
            n_train_samples=1000,
            n_infer_samples=n_infer_samples,
            peak_gpu_mem_bytes=peak_gpu,
            device="cpu",
        )

    def test_empty_returns_empty(self):
        assert aggregate_efficiency_records([]) == {}

    def test_single_record_mean_equals_value(self):
        rec = self._rec(train_time_s=3.0, infer_time_s=0.01)
        agg = aggregate_efficiency_records([rec])
        assert abs(agg["mean_train_time_s"] - 3.0) < 1e-6
        assert abs(agg["median_train_time_s"] - 3.0) < 1e-6

    def test_mean_train_time(self):
        recs = [self._rec(t, 0.01) for t in [1.0, 2.0, 3.0]]
        agg = aggregate_efficiency_records(recs)
        assert abs(agg["mean_train_time_s"] - 2.0) < 1e-6

    def test_median_train_time(self):
        recs = [self._rec(t, 0.01) for t in [1.0, 2.0, 100.0]]
        agg = aggregate_efficiency_records(recs)
        assert abs(agg["median_train_time_s"] - 2.0) < 1e-6

    def test_mean_infer_per_sample(self):
        recs = [self._rec(1.0, infer_time_s=i, n_infer_samples=100)
                for i in [0.01, 0.03]]  # per sample: 1e-4, 3e-4
        agg = aggregate_efficiency_records(recs)
        assert abs(agg["mean_infer_time_s_per_sample"] - 2e-4) < 1e-10

    def test_max_peak_gpu_mem(self):
        recs = [self._rec(1.0, 0.01, peak_gpu=g) for g in [100, 500, 200]]
        agg = aggregate_efficiency_records(recs)
        assert agg["max_peak_gpu_mem_bytes"] == 500

    def test_n_params_from_first_record(self):
        recs = [self._rec(1.0, 0.01) for _ in range(3)]
        agg = aggregate_efficiency_records(recs)
        assert agg["n_params"] == 256

    def test_n_records_field(self):
        recs = [self._rec(1.0, 0.01) for _ in range(5)]
        agg = aggregate_efficiency_records(recs)
        assert agg["n_records"] == 5

    def test_device_from_first_record(self):
        recs = [self._rec(1.0, 0.01) for _ in range(2)]
        agg = aggregate_efficiency_records(recs)
        assert agg["device"] == "cpu"


# ---------------------------------------------------------------------------
# CUDA helpers (CPU path — no GPU required)
# ---------------------------------------------------------------------------

class TestCudaHelpersOnCPU:
    def test_get_peak_gpu_memory_returns_zero_on_cpu(self):
        assert get_peak_gpu_memory_bytes("cpu") == 0

    def test_reset_peak_gpu_memory_noop_on_cpu(self):
        reset_peak_gpu_memory("cpu")  # should not raise

    def test_sync_cuda_noop_on_cpu(self):
        sync_cuda_if_needed("cpu")  # should not raise


# ---------------------------------------------------------------------------
# XGBoostWrapper efficiency attributes
# ---------------------------------------------------------------------------

class TestXGBoostWrapperEfficiencyAttrs:
    @pytest.fixture
    def fitted_model(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10)).astype(np.float32)
        y = rng.integers(0, 2, size=200)
        model = XGBoostWrapper(n_estimators=5, max_depth=3, seed=42, device="cpu")
        model.fit(X, y)
        return model, X, y

    def test_train_time_s_is_positive(self, fitted_model):
        model, _, _ = fitted_model
        assert model.train_time_s > 0.0

    def test_n_params_is_positive_int(self, fitted_model):
        model, _, _ = fitted_model
        assert isinstance(model.n_params, int)
        assert model.n_params > 0

    def test_infer_time_s_set_after_predict_proba(self, fitted_model):
        model, X, _ = fitted_model
        model.predict_proba(X)
        assert model.infer_time_s > 0.0

    def test_predict_proba_updates_infer_time(self, fitted_model):
        model, X, _ = fitted_model
        model.predict_proba(X)
        t1 = model.infer_time_s
        model.predict_proba(X)
        t2 = model.infer_time_s
        # infer_time is re-measured each call; should be finite and positive
        assert t1 > 0 and t2 > 0

    def test_efficiency_record_builds_from_xgboost(self, fitted_model):
        model, X, _ = fitted_model
        model.predict_proba(X)
        rec = build_efficiency_record(
            n_params=model.n_params,
            train_time_s=model.train_time_s,
            infer_time_s=model.infer_time_s,
            n_train_samples=200,
            n_infer_samples=len(X),
            peak_gpu_mem_bytes=0,
            device="cpu",
        )
        assert rec["n_params"] > 0
        assert rec["train_time_s"] > 0
        assert rec["infer_time_s_per_sample"] is not None
        assert rec["infer_time_s_per_sample"] > 0


# ---------------------------------------------------------------------------
# measure_pytorch_inference — synchronization contract
# ---------------------------------------------------------------------------

def _make_fake_predict_fn(y_true, y_pred, y_proba):
    """Return a callable that mimics predict_pytorch's signature and return type."""
    def _fn(model, loader, device):
        return y_true, y_pred, y_proba
    return _fn


class TestMeasurePytorchInference:
    _y_true  = np.array([0, 1, 0, 1])
    _y_pred  = np.array([0, 1, 1, 1])
    _y_proba = np.array([0.1, 0.9, 0.6, 0.8])

    def _call(self, device="cpu", predict_fn=None):
        if predict_fn is None:
            predict_fn = _make_fake_predict_fn(self._y_true, self._y_pred, self._y_proba)
        model  = MagicMock(spec=nn.Module)
        loader = MagicMock()
        return measure_pytorch_inference(predict_fn, model, loader, device)

    # ── output contract ────────────────────────────────────────────────────

    def test_returns_four_tuple(self):
        result = self._call()
        assert len(result) == 4

    def test_y_true_y_pred_y_proba_pass_through(self):
        y_true, y_pred, y_proba, _ = self._call()
        np.testing.assert_array_equal(y_true,  self._y_true)
        np.testing.assert_array_equal(y_pred,  self._y_pred)
        np.testing.assert_array_equal(y_proba, self._y_proba)

    def test_elapsed_is_positive_float(self):
        _, _, _, elapsed = self._call()
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    # ── synchronization contract on CPU ───────────────────────────────────

    def test_sync_called_before_and_after_on_cpu(self):
        """sync_cuda_if_needed must be called exactly twice (before + after)."""
        target = "lift_nids.evaluation.efficiency.sync_cuda_if_needed"
        with patch(target) as mock_sync:
            predict_fn = _make_fake_predict_fn(self._y_true, self._y_pred, self._y_proba)
            model  = MagicMock(spec=nn.Module)
            loader = MagicMock()
            measure_pytorch_inference(predict_fn, model, loader, "cpu")

        assert mock_sync.call_count == 2
        assert mock_sync.call_args_list == [call("cpu"), call("cpu")]

    def test_sync_called_before_inference_starts(self):
        """First sync must fire before predict_fn is invoked."""
        call_order: list[str] = []
        target = "lift_nids.evaluation.efficiency.sync_cuda_if_needed"

        def _tracking_sync(device):
            call_order.append("sync")

        def _tracking_predict(model, loader, device):
            call_order.append("predict")
            return self._y_true, self._y_pred, self._y_proba

        with patch(target, side_effect=_tracking_sync):
            model  = MagicMock(spec=nn.Module)
            loader = MagicMock()
            measure_pytorch_inference(_tracking_predict, model, loader, "cpu")

        assert call_order == ["sync", "predict", "sync"]

    def test_sync_is_noop_on_cpu_no_error(self):
        """CPU path must not raise, even without a CUDA device."""
        _, _, _, elapsed = self._call(device="cpu")
        assert elapsed >= 0.0

    # ── predict_fn receives correct arguments ──────────────────────────────

    def test_predict_fn_receives_model_loader_device(self):
        received: dict = {}

        def _capturing_predict(model, loader, device):
            received["model"]  = model
            received["loader"] = loader
            received["device"] = device
            return self._y_true, self._y_pred, self._y_proba

        model  = MagicMock(spec=nn.Module)
        loader = MagicMock()
        measure_pytorch_inference(_capturing_predict, model, loader, "cpu")

        assert received["model"]  is model
        assert received["loader"] is loader
        assert received["device"] == "cpu"
