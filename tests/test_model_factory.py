"""Tests for build_torch_model — construct a PyTorch model from a params dict.

The factory replaces the ``_suggest_pytorch_model(_MockTrial(params), ...)``
idiom used to rebuild a model from Optuna ``best_params`` at refit time.
"""

from __future__ import annotations

import pytest
import torch

from lift_nids.models.cnn_bilstm import CNNBiLSTM
from lift_nids.models.factory import build_torch_model
from lift_nids.models.mlp_baseline import MLPBaseline
from lift_nids.models.transformer_light import LightTransformer

N_FEATURES = 20
WINDOW = 8

_PARAMS: dict[str, dict] = {
    "mlp": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_config_idx": 1,
        "dropout": 0.3,
        "batch_norm": True,
        "batch_size": 512,
    },
    "cnn_bilstm": {
        "lr": 2e-3,
        "weight_decay": 5e-4,
        "channel_config_idx": 2,
        "lstm_hidden_size": 96,
        "dropout": 0.25,
        "batch_size": 128,
    },
    "transformer": {
        "lr": 5e-4,
        "weight_decay": 9e-4,
        "dropout": 0.2,
        "head_dropout": 0.15,
        "batch_size": 256,
    },
}


class _MockTrial:
    """Replays stored params through the ``trial.suggest_*`` interface.

    Mirrors the shim the runners used before the factory existed, so the
    equivalence test can drive the original ``_suggest_pytorch_model`` path.
    """

    def __init__(self, params: dict) -> None:
        self._p = params

    def suggest_float(self, name, *_a, **_kw):
        return self._p[name]

    def suggest_int(self, name, *_a, **_kw):
        return self._p[name]

    def suggest_categorical(self, name, *_a, **_kw):
        return self._p[name]


def test_build_mlp_returns_model_and_training_params():
    params = _PARAMS["mlp"]
    model, lr, weight_decay, batch_size = build_torch_model("mlp", params, n_features=N_FEATURES)

    assert isinstance(model, MLPBaseline)
    assert lr == params["lr"]
    assert weight_decay == params["weight_decay"]
    assert batch_size == params["batch_size"]

    out = model(torch.randn(4, N_FEATURES))
    assert out.shape == (4, 2)


def test_build_cnn_bilstm_returns_sequence_model():
    params = _PARAMS["cnn_bilstm"]
    model, lr, weight_decay, batch_size = build_torch_model(
        "cnn_bilstm", params, n_features=N_FEATURES
    )

    assert isinstance(model, CNNBiLSTM)
    assert (lr, weight_decay, batch_size) == (
        params["lr"],
        params["weight_decay"],
        params["batch_size"],
    )
    out = model(torch.randn(4, WINDOW, N_FEATURES))
    assert out.shape == (4, 2)


def test_build_transformer_returns_sequence_model():
    params = _PARAMS["transformer"]
    model, lr, weight_decay, batch_size = build_torch_model(
        "transformer", params, n_features=N_FEATURES
    )

    assert isinstance(model, LightTransformer)
    assert (lr, weight_decay, batch_size) == (
        params["lr"],
        params["weight_decay"],
        params["batch_size"],
    )
    out = model(torch.randn(4, WINDOW, N_FEATURES))
    assert out.shape == (4, 2)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_torch_model("random_forest", {"lr": 1, "weight_decay": 0, "batch_size": 1}, 10)


@pytest.mark.parametrize("family", ["mlp", "cnn_bilstm", "transformer"])
def test_factory_matches_suggest_pytorch_model(family: str):
    """build_torch_model must reproduce the original construction bit-for-bit.

    Guards the refactor: under the same seed, the factory and the legacy
    ``_suggest_pytorch_model`` path must produce identical weights and identical
    (lr, weight_decay, batch_size).
    """
    from lift_nids.training.optuna_search import _suggest_pytorch_model

    params = _PARAMS[family]

    torch.manual_seed(1234)
    ref_model, ref_lr, ref_wd, ref_bs = _suggest_pytorch_model(
        _MockTrial(params), family, N_FEATURES, 2
    )

    torch.manual_seed(1234)
    new_model, new_lr, new_wd, new_bs = build_torch_model(family, params, N_FEATURES, 2)

    assert (new_lr, new_wd, new_bs) == (ref_lr, ref_wd, ref_bs)

    ref_state = ref_model.state_dict()
    new_state = new_model.state_dict()
    assert ref_state.keys() == new_state.keys()
    for key in ref_state:
        assert torch.equal(new_state[key], ref_state[key]), f"weight mismatch at {key}"
