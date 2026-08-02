"""Tests for all four model families using synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lift_nids.models.cnn_bilstm import CNNBiLSTM
from lift_nids.models.mlp_baseline import MLPBaseline
from lift_nids.models.transformer_light import (
    D_MODEL,
    N_HEADS,
    NUM_LAYERS,
    LightTransformer,
)
from lift_nids.models.xgboost_wrapper import XGBoostWrapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 8
N_FEATURES = 32
WIN = 16
N_CLASSES = 2

RNG = np.random.default_rng(42)


@pytest.fixture()
def tabular_batch() -> tuple[np.ndarray, np.ndarray]:
    X = RNG.standard_normal((BATCH, N_FEATURES)).astype(np.float32)
    y = RNG.integers(0, N_CLASSES, size=BATCH)
    return X, y


@pytest.fixture()
def sequence_tensor() -> torch.Tensor:
    return torch.randn(BATCH, WIN, N_FEATURES)


# ---------------------------------------------------------------------------
# XGBoostWrapper
# ---------------------------------------------------------------------------


class TestXGBoostWrapper:
    def test_fit_predict_shapes(self, tabular_batch):
        X, y = tabular_batch
        model = XGBoostWrapper(n_estimators=10, seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (BATCH,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self, tabular_batch):
        X, y = tabular_batch
        model = XGBoostWrapper(n_estimators=10, seed=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (BATCH,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_fit_with_val_set(self, tabular_batch):
        X, y = tabular_batch
        model = XGBoostWrapper(n_estimators=10, seed=42)
        model.fit(X, y, X_val=X, y_val=y)   # same data — just checks no crash
        assert model.predict(X).shape == (BATCH,)

    def test_save_load(self, tabular_batch, tmp_path):
        X, y = tabular_batch
        model = XGBoostWrapper(n_estimators=10, seed=42)
        model.fit(X, y)
        orig_preds = model.predict(X)

        path = tmp_path / "xgb.json"
        model.save(path)
        loaded = XGBoostWrapper.load(path)
        assert np.array_equal(loaded.predict(X), orig_preds)

    def test_from_dict_constructor(self, tabular_batch):
        X, y = tabular_batch
        params = {"n_estimators": 10, "max_depth": 4, "seed": 42}
        model = XGBoostWrapper.from_dict(params)
        model.fit(X, y)
        assert model.predict(X).shape == (BATCH,)

    def test_from_dict_unknown_keys_raise(self):
        with pytest.raises(TypeError):
            XGBoostWrapper.from_dict({"n_estimators": 10, "nonexistent_param": 99})

    def test_train_time_recorded(self, tabular_batch):
        X, y = tabular_batch
        model = XGBoostWrapper(n_estimators=10, seed=42)
        model.fit(X, y)
        assert hasattr(model, "train_time_s")
        assert model.train_time_s > 0

    def test_infer_time_recorded_after_predict(self, tabular_batch):
        X, y = tabular_batch
        model = XGBoostWrapper(n_estimators=10, seed=42)
        model.fit(X, y)
        model.predict(X)
        assert hasattr(model, "infer_time_s")
        assert model.infer_time_s > 0

    def test_infer_time_recorded_after_predict_proba(self, tabular_batch):
        X, y = tabular_batch
        model = XGBoostWrapper(n_estimators=10, seed=42)
        model.fit(X, y)
        model.predict_proba(X)
        assert hasattr(model, "infer_time_s")
        assert model.infer_time_s > 0

    def test_n_params_after_fit(self, tabular_batch):
        X, y = tabular_batch
        n_estimators, max_depth = 10, 5
        model = XGBoostWrapper(n_estimators=n_estimators, max_depth=max_depth, seed=42)
        model.fit(X, y)
        assert hasattr(model, "n_params")
        assert model.n_params == n_estimators * max_depth


# ---------------------------------------------------------------------------
# MLPBaseline
# ---------------------------------------------------------------------------


class TestMLPBaseline:
    def test_forward_shape(self, sequence_tensor):
        x = sequence_tensor[:, 0, :]  # tabular: (batch, n_features)
        model = MLPBaseline(N_FEATURES, N_CLASSES)
        out = model(x)
        assert out.shape == (BATCH, N_CLASSES)

    def test_default_hidden_dims(self):
        model = MLPBaseline(N_FEATURES, N_CLASSES)
        # Verify 3-layer default: Linear -> BN -> ReLU -> Dropout x3
        n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        assert n_linear == 4  # 3 hidden + 1 output

    def test_custom_hidden_dims(self):
        model = MLPBaseline(N_FEATURES, N_CLASSES, hidden_dims=[64, 32])
        n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        assert n_linear == 3  # 2 hidden + 1 output

    def test_no_batch_norm(self):
        model = MLPBaseline(N_FEATURES, N_CLASSES, batch_norm=False)
        n_bn = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d))
        assert n_bn == 0

    def test_gradients_flow(self):
        model = MLPBaseline(N_FEATURES, N_CLASSES)
        x = torch.randn(BATCH, N_FEATURES, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# CNNBiLSTM
# ---------------------------------------------------------------------------


class TestCNNBiLSTM:
    def test_forward_shape(self, sequence_tensor):
        model = CNNBiLSTM(N_FEATURES, N_CLASSES)
        out = model(sequence_tensor)
        assert out.shape == (BATCH, N_CLASSES)

    def test_custom_channels(self, sequence_tensor):
        model = CNNBiLSTM(N_FEATURES, N_CLASSES, conv_channels=[32], kernel_sizes=[3])
        out = model(sequence_tensor)
        assert out.shape == (BATCH, N_CLASSES)

    def test_mismatched_channels_raises(self):
        with pytest.raises(AssertionError):
            CNNBiLSTM(N_FEATURES, N_CLASSES, conv_channels=[64, 128], kernel_sizes=[3])

    def test_gradients_flow(self, sequence_tensor):
        model = CNNBiLSTM(N_FEATURES, N_CLASSES)
        out = model(sequence_tensor)
        out.sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_window_sizes(self):
        model = CNNBiLSTM(N_FEATURES, N_CLASSES)
        for win in [8, 16, 32]:
            x = torch.randn(BATCH, win, N_FEATURES)
            assert model(x).shape == (BATCH, N_CLASSES)


# ---------------------------------------------------------------------------
# LightTransformer
# ---------------------------------------------------------------------------


class TestLightTransformer:
    def test_forward_shape(self, sequence_tensor):
        model = LightTransformer(N_FEATURES, N_CLASSES)
        out = model(sequence_tensor)
        assert out.shape == (BATCH, N_CLASSES)

    def test_fixed_architecture_constants(self):
        assert D_MODEL == 128
        assert N_HEADS == 2
        assert NUM_LAYERS == 2

    def test_default_uses_im09_config(self):
        model = LightTransformer(N_FEATURES, N_CLASSES)
        assert model.input_proj.out_features == D_MODEL
        n_layers = len(model.layers)
        assert n_layers == NUM_LAYERS

    def test_causal_mask_shape(self, sequence_tensor):
        model = LightTransformer(N_FEATURES, N_CLASSES)
        mask = model._causal_mask(WIN, sequence_tensor.device)
        assert mask.shape == (WIN, WIN)
        # Upper triangle should be True (masked), diagonal and below False
        assert mask[0, 0] is False or not mask[0, 0].item()
        assert mask[0, 1].item() is True

    def test_last_token_head(self, sequence_tensor):
        model = LightTransformer(N_FEATURES, N_CLASSES)
        out = model(sequence_tensor)
        assert out.shape == (BATCH, N_CLASSES), "Last Token head must produce (batch, n_classes)"

    def test_gradients_flow(self, sequence_tensor):
        model = LightTransformer(N_FEATURES, N_CLASSES)
        out = model(sequence_tensor)
        out.sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_window_sizes(self):
        model = LightTransformer(N_FEATURES, N_CLASSES)
        for win in [8, 16, 32]:
            x = torch.randn(BATCH, win, N_FEATURES)
            assert model(x).shape == (BATCH, N_CLASSES)

    def test_multiclass_output(self, sequence_tensor):
        model = LightTransformer(N_FEATURES, n_classes=5)
        out = model(sequence_tensor)
        assert out.shape == (BATCH, 5)
