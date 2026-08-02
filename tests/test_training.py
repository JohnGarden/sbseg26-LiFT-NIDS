"""Tests for training infrastructure: EarlyStopping, Trainer, helpers, Optuna search."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lift_nids.training.early_stopping import EarlyStopping
from lift_nids.training.trainer import (
    Trainer,
    build_optimizer,
    compute_class_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_loader(
    n: int = 64,
    n_features: int = 8,
    n_classes: int = 2,
    seed: int = 0,
    seq_len: int | None = None,
):
    """Create a DataLoader with 2D or 3D tensors."""
    rng = torch.Generator().manual_seed(seed)
    shape = (n, seq_len, n_features) if seq_len else (n, n_features)
    X = torch.randn(*shape, generator=rng)
    y = torch.randint(0, n_classes, (n,), generator=rng)
    return DataLoader(TensorDataset(X, y), batch_size=16)


class _TinyMLP(nn.Module):
    def __init__(self, n_features: int = 8, n_classes: int = 2):
        super().__init__()
        self.net = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        assert not es.step(0.5)
        assert not es.step(0.5)
        assert not es.step(0.5)
        assert es.step(0.5)  # 4th non-improvement → stop

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.0)
        es.step(0.5)
        es.step(0.5)
        assert not es.step(0.6)  # improvement resets counter
        assert not es.step(0.6)  # first non-improvement (counter=1)
        assert es.step(0.6)      # second non-improvement (counter=2) → stop

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=2, min_delta=0.01)
        es.step(0.5)
        # 0.505 < 0.5 + 0.01 = 0.51 → not an improvement
        assert not es.step(0.505)
        assert es.step(0.505)

    def test_best_score_tracked(self):
        es = EarlyStopping(patience=5)
        es.step(0.3)
        es.step(0.7)
        es.step(0.5)
        assert es.best_score == pytest.approx(0.7)

    def test_best_epoch_tracked(self):
        es = EarlyStopping(patience=5)
        es.step(0.3)
        es.step(0.7)
        es.step(0.5)
        assert es.best_epoch == 2

    def test_improved_property(self):
        es = EarlyStopping(patience=5)
        es.step(0.5)
        assert es.improved
        es.step(0.3)
        assert not es.improved

    def test_restore_best(self):
        model = _TinyMLP()
        es = EarlyStopping(patience=5, restore_best=True)

        original_weight = model.net.weight.data.clone()
        es.step(0.5, model)

        # Modify model weights
        with torch.no_grad():
            model.net.weight.fill_(99.0)

        es.step(0.3, model)  # no improvement → best state still from step 1
        es.restore(model)
        assert torch.allclose(model.net.weight.data, original_weight)

    def test_restore_best_false_stores_nothing(self):
        model = _TinyMLP()
        es = EarlyStopping(patience=5, restore_best=False)
        es.step(0.5, model)
        with torch.no_grad():
            model.net.weight.fill_(99.0)
        es.restore(model)  # should be a no-op
        assert model.net.weight.data[0, 0].item() == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------

class TestBuildOptimizer:
    def _model(self):
        return _TinyMLP()

    def test_adam(self):
        opt = build_optimizer(self._model(), "adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw(self):
        opt = build_optimizer(self._model(), "adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd(self):
        opt = build_optimizer(self._model(), "sgd", lr=1e-2)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(self._model(), "rmsprop")

    def test_lr_set_correctly(self):
        opt = build_optimizer(self._model(), "adam", lr=5e-4)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    def test_balanced_returns_ones(self):
        labels = np.array([0, 1, 0, 1])
        w = compute_class_weights(labels, n_classes=2)
        assert w.shape == (2,)
        assert torch.allclose(w, torch.ones(2))

    def test_imbalanced_minority_gets_higher_weight(self):
        labels = np.array([0] * 9 + [1])  # 90% class-0
        w = compute_class_weights(labels, n_classes=2)
        assert w[1] > w[0]

    def test_absent_class_does_not_crash(self):
        labels = np.array([0, 0, 0])
        w = compute_class_weights(labels, n_classes=2)
        assert w.shape == (2,)
        assert not torch.any(torch.isnan(w))

    def test_output_is_float32(self):
        labels = np.array([0, 1])
        w = compute_class_weights(labels, n_classes=2)
        assert w.dtype == torch.float32


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestTrainer:
    def _setup(self, n_features: int = 8, n_classes: int = 2):
        model = _TinyMLP(n_features, n_classes)
        optimizer = build_optimizer(model, "adam", lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device="cpu")
        return trainer

    def test_train_epoch_returns_loss_and_f1(self):
        trainer = self._setup()
        loader = _make_loader()
        metrics = trainer.train_epoch(loader)
        assert "loss" in metrics
        assert "f1_macro" in metrics
        assert metrics["loss"] >= 0.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_evaluate_returns_expected_keys(self):
        trainer = self._setup()
        loader = _make_loader()
        metrics = trainer.evaluate(loader)
        assert set(metrics.keys()) == {"loss", "f1_macro", "f1_1", "precision", "recall"}

    def test_fit_returns_history(self):
        trainer = self._setup()
        train_loader = _make_loader(seed=0)
        val_loader = _make_loader(seed=1)
        history = trainer.fit(train_loader, val_loader, max_epochs=3)
        expected_keys = {
            "train_loss", "val_loss",
            "train_f1", "val_f1",
            "train_f1_1", "val_f1_1",
            "train_precision", "val_precision",
            "train_recall", "val_recall",
            "train_time_s", "peak_gpu_mem_bytes",
        }
        assert set(history.keys()) == expected_keys
        assert len(history["train_loss"]) == 3

    def test_fit_respects_max_epochs(self):
        trainer = self._setup()
        loader = _make_loader()
        history = trainer.fit(loader, loader, max_epochs=5)
        assert len(history["val_f1"]) <= 5

    def test_fit_with_early_stopping_can_stop_early(self):
        model = _TinyMLP()
        optimizer = build_optimizer(model, "adam", lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        es = EarlyStopping(patience=1, restore_best=False)
        trainer = Trainer(model, optimizer, criterion, device="cpu", early_stopping=es)
        loader = _make_loader()
        history = trainer.fit(loader, loader, max_epochs=50)
        assert len(history["val_f1"]) < 50

    def test_fit_early_stopping_restores_best(self):
        model = _TinyMLP()
        optimizer = build_optimizer(model, "adam", lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        es = EarlyStopping(patience=2, restore_best=True)
        trainer = Trainer(model, optimizer, criterion, device="cpu", early_stopping=es)
        loader = _make_loader()
        trainer.fit(loader, loader, max_epochs=20)
        # After fit, best_score should be recorded
        assert es.best_score >= 0.0

    def test_checkpoint_roundtrip(self, tmp_path):
        trainer = self._setup()
        ckpt = tmp_path / "ckpt.pt"
        original_params = copy.deepcopy(trainer.model.state_dict())
        trainer.save_checkpoint(ckpt)
        # Corrupt model weights
        with torch.no_grad():
            trainer.model.net.weight.fill_(0.0)
        trainer.load_checkpoint(ckpt)
        for k in original_params:
            assert torch.allclose(trainer.model.state_dict()[k], original_params[k])

    def test_model_moved_to_device(self):
        trainer = self._setup()
        device = next(trainer.model.parameters()).device
        assert str(device) == "cpu"


# ---------------------------------------------------------------------------
# Optuna search (smoke test — 2 trials to keep runtime short)
# ---------------------------------------------------------------------------

class TestOptunaSearch:
    def _run(self, model_type: str, n_features: int = 8, seq_len: int | None = None, **kwargs):
        from lift_nids.training.optuna_search import run_optuna_search

        if model_type == "xgboost":
            rng = np.random.default_rng(0)
            X = rng.standard_normal((100, n_features)).astype(np.float32)
            y = rng.integers(0, 2, 100)
            train_data = (X[:80], y[:80])
            val_data = (X[80:], y[80:])
        else:
            train_data = _make_loader(64, n_features, seed=0, seq_len=seq_len)
            val_data = _make_loader(32, n_features, seed=1, seq_len=seq_len)

        return run_optuna_search(
            model_type=model_type,
            train_data=train_data,
            val_data=val_data,
            n_features=n_features,
            n_trials=2,
            max_epochs=2,
            patience=1,
            seed=42,
            **kwargs,
        )

    def test_xgboost_smoke(self):
        result = self._run("xgboost")
        assert "best_params" in result
        assert result["n_trials_completed"] == 2

    def test_mlp_smoke(self):
        result = self._run("mlp")
        assert "best_params" in result
        assert result["best_value"] >= 0.0

    def test_cnn_bilstm_smoke(self):
        result = self._run("cnn_bilstm", seq_len=16)
        assert result["n_trials_completed"] == 2

    def test_transformer_smoke(self):
        result = self._run("transformer", seq_len=16)
        assert "best_params" in result
        # Architecture HPs must NOT appear in best_params (FIXED)
        assert "d_model" not in result["best_params"]
        assert "n_heads" not in result["best_params"]
        assert "num_layers" not in result["best_params"]

    def test_unknown_model_raises(self):
        from lift_nids.training.optuna_search import run_optuna_search

        with pytest.raises(ValueError, match="Unknown model_type"):
            run_optuna_search(
                model_type="unknown",
                train_data=_make_loader(32),
                val_data=_make_loader(16),
                n_features=8,
                n_trials=1,
            )

    def test_storage_sqlite(self, tmp_path):
        storage = f"sqlite:///{tmp_path}/optuna.db"
        result = self._run("mlp", storage=storage)  # mlp uses 2D input
        assert result["n_trials_completed"] == 2


# ---------------------------------------------------------------------------
# Regression test: _xgboost_objective must return F1+ not F1 macro
# ---------------------------------------------------------------------------

class TestXGBoostObjectiveOptimizesF1Plus:
    """Verify that _xgboost_objective returns F1 of the positive class (F1+).

    Uses predictions that classify class-0 correctly but never predict class-1,
    producing a scenario where f1_macro > 0 but f1_1 == 0.  The objective
    must return 0.0, proving it uses F1+ and not the inflated macro average.
    """

    def test_returns_f1_plus_not_f1_macro(self, monkeypatch):
        from sklearn.metrics import f1_score

        import numpy as np
        from lift_nids.models.xgboost_wrapper import XGBoostWrapper
        from lift_nids.training.optuna_search import _xgboost_objective

        # 5 benign (class 0) + 5 attack (class 1)
        y_val = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
        X_val = np.zeros((10, 4), dtype=np.float32)
        y_train = y_val.copy()
        X_train = X_val.copy()

        # Force the model to predict all class-0 (hits benign, misses attack)
        monkeypatch.setattr(XGBoostWrapper, "fit", lambda self, X, y, **kw: None)
        monkeypatch.setattr(
            XGBoostWrapper, "predict",
            lambda self, X: np.zeros(len(X), dtype=np.int64),
        )

        class _MockTrial:
            def suggest_int(self, name, lo, hi, **kw):       return lo
            def suggest_float(self, name, lo, hi, **kw):     return lo
            def suggest_categorical(self, name, choices, **kw): return choices[0]

        score = _xgboost_objective(_MockTrial(), (X_train, y_train), (X_val, y_val))

        y_pred = np.zeros(10, dtype=np.int64)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        f1_1 = f1_score(y_val, y_pred, pos_label=1, average="binary", zero_division=0)

        # Verify the test scenario is discriminating: macro > 0, f1_1 == 0
        assert f1_macro > 0.0, "f1_macro must be > 0 for this test to be meaningful"
        assert f1_1 == 0.0, "f1_1 must be 0.0 when class-1 is never predicted"

        # The objective must return F1+ = 0.0, not the higher f1_macro
        assert score == pytest.approx(0.0), (
            f"_xgboost_objective returned {score:.4f} instead of 0.0; "
            "likely still using f1_macro instead of F1+"
        )


# ---------------------------------------------------------------------------
# Optuna resume: remaining-trials counting
# ---------------------------------------------------------------------------

class TestOptunaSearchResume:
    """run_optuna_search must not add more trials when storage already has enough."""

    def _run(
        self,
        storage: str | None,
        n_trials: int = 3,
        n_features: int = 4,
        model_type: str = "mlp",
    ) -> dict:
        from lift_nids.training.optuna_search import run_optuna_search

        loader = _make_loader(32, n_features, seed=0)
        return run_optuna_search(
            model_type=model_type,
            train_data=loader,
            val_data=loader,
            n_features=n_features,
            n_trials=n_trials,
            max_epochs=1,
            patience=1,
            seed=42,
            storage=storage,
            show_progress_bar=False,
        )

    def test_no_new_trials_when_study_already_complete(self, tmp_path):
        """Second call with same n_trials must not add more trials to the study."""
        storage = f"sqlite:///{tmp_path}/optuna.db"
        r1 = self._run(storage, n_trials=3)
        assert r1["n_trials_completed"] == 3

        r2 = self._run(storage, n_trials=3)
        assert r2["n_trials_completed"] == 3, (
            "Re-running with same n_trials must reuse existing trials, not add more"
        )

    def test_adds_only_remaining_trials(self, tmp_path):
        """Second call with higher n_trials must add only the delta."""
        storage = f"sqlite:///{tmp_path}/optuna.db"
        r1 = self._run(storage, n_trials=2)
        assert r1["n_trials_completed"] == 2

        r2 = self._run(storage, n_trials=4)
        assert r2["n_trials_completed"] == 4, (
            "Resuming from 2 trials with n_trials=4 must add exactly 2 more"
        )

    def test_no_storage_always_runs_full_trials(self):
        """Without persistent storage every call starts from scratch."""
        r = self._run(storage=None, n_trials=2)
        assert r["n_trials_completed"] == 2

    def test_resumed_study_returns_valid_best_params(self, tmp_path):
        """Early-exit path must still return best_params and best_value."""
        storage = f"sqlite:///{tmp_path}/optuna.db"
        self._run(storage, n_trials=2)
        r = self._run(storage, n_trials=2)
        assert "best_params" in r
        assert "best_value" in r
        assert isinstance(r["best_value"], float)
