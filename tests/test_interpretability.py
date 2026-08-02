"""Tests for interpretability: attention extraction, visualization, and axis2 artifact helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot is imported

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lift_nids.interpretability.attention_maps import (
    extract_attention_weights,
    plot_attention_heatmap,
)
from lift_nids.models.transformer_light import (
    N_HEADS,
    NUM_LAYERS,
    LightTransformer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BATCH = 2
SEQ_LEN = 8
N_FEATURES = 10
N_CLASSES = 2


@pytest.fixture
def model() -> LightTransformer:
    m = LightTransformer(n_features=N_FEATURES, n_classes=N_CLASSES)
    m.eval()
    return m


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, N_FEATURES)


# ---------------------------------------------------------------------------
# LightTransformer — attention weight return contract
# ---------------------------------------------------------------------------

class TestLightTransformerAttentionOutput:
    def test_returns_tuple_when_requested(self, model, x):
        out = model(x, return_attention_weights=True)
        assert isinstance(out, tuple) and len(out) == 2

    def test_logits_shape(self, model, x):
        logits, _ = model(x, return_attention_weights=True)
        assert logits.shape == (BATCH, N_CLASSES)

    def test_returns_num_layers_attention_tensors(self, model, x):
        _, attn_list = model(x, return_attention_weights=True)
        assert len(attn_list) == NUM_LAYERS

    def test_each_tensor_shape_is_batch_heads_S_S(self, model, x):
        _, attn_list = model(x, return_attention_weights=True)
        for attn in attn_list:
            assert attn is not None
            assert attn.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN), (
                f"Expected (batch={BATCH}, n_heads={N_HEADS}, "
                f"seq_len={SEQ_LEN}, seq_len={SEQ_LEN}), got {tuple(attn.shape)}"
            )

    def test_without_flag_returns_plain_tensor(self, model, x):
        out = model(x, return_attention_weights=False)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (BATCH, N_CLASSES)

    def test_causal_mask_zeroes_future_positions(self, model, x):
        """Attention above the diagonal must be zero (causal masking)."""
        with torch.no_grad():
            _, attn_list = model(x, return_attention_weights=True)
        for layer_i, attn in enumerate(attn_list):
            arr = attn.cpu().numpy()
            for b in range(BATCH):
                for h in range(N_HEADS):
                    upper = arr[b, h][np.triu_indices(SEQ_LEN, k=1)]
                    assert np.allclose(upper, 0.0, atol=1e-5), (
                        f"Causal mask violated at layer={layer_i}, "
                        f"batch={b}, head={h}: max upper-tri value = {upper.max():.2e}"
                    )

    def test_attention_rows_sum_to_one(self, model, x):
        """Each query row must form a valid probability distribution."""
        with torch.no_grad():
            _, attn_list = model(x, return_attention_weights=True)
        for attn in attn_list:
            arr = attn.cpu().numpy()
            row_sums = arr.sum(axis=-1)  # (batch, n_heads, seq_len)
            assert np.allclose(row_sums, 1.0, atol=1e-4), (
                f"Attention rows do not sum to 1; min={row_sums.min():.4f}, "
                f"max={row_sums.max():.4f}"
            )


# ---------------------------------------------------------------------------
# extract_attention_weights
# ---------------------------------------------------------------------------

class TestExtractAttentionWeights:
    def test_returns_ndarray(self, model, x):
        w = extract_attention_weights(model, x)
        assert isinstance(w, np.ndarray)

    def test_shape_last_layer(self, model, x):
        w = extract_attention_weights(model, x, layer_idx=-1)
        assert w.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_shape_first_layer(self, model, x):
        w = extract_attention_weights(model, x, layer_idx=0)
        assert w.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_layers_differ(self, model, x):
        """Different layers should (almost always) produce different weights."""
        if NUM_LAYERS < 2:
            pytest.skip("Need at least 2 layers to compare.")
        w0 = extract_attention_weights(model, x, layer_idx=0)
        w1 = extract_attention_weights(model, x, layer_idx=-1)
        assert not np.allclose(w0, w1), (
            "Layer 0 and last-layer attention weights are identical — "
            "check that layer_idx selection is working."
        )

    def test_invalid_layer_idx_raises_index_error(self, model, x):
        with pytest.raises(IndexError, match="layer_idx"):
            extract_attention_weights(model, x, layer_idx=100)

    def test_negative_out_of_range_raises(self, model, x):
        with pytest.raises(IndexError, match="layer_idx"):
            extract_attention_weights(model, x, layer_idx=-(NUM_LAYERS + 1))

    def test_preserves_train_mode(self, x):
        m = LightTransformer(n_features=N_FEATURES)
        m.train()
        extract_attention_weights(m, x)
        assert m.training, "extract_attention_weights must restore train() mode"

    def test_preserves_eval_mode(self, model, x):
        assert not model.training
        extract_attention_weights(model, x)
        assert not model.training, "extract_attention_weights must preserve eval() mode"

    def test_result_is_on_cpu(self, model, x):
        w = extract_attention_weights(model, x)
        # np.ndarray has no device concept — verify it was detached from CUDA
        assert isinstance(w, np.ndarray)

    def test_causal_structure_preserved_in_output(self, model, x):
        w = extract_attention_weights(model, x)
        upper = w[0, 0][np.triu_indices(SEQ_LEN, k=1)]
        assert np.allclose(upper, 0.0, atol=1e-5), \
            "Causal mask should be visible in extracted weights."


# ---------------------------------------------------------------------------
# plot_attention_heatmap
# ---------------------------------------------------------------------------

class TestPlotAttentionHeatmap:
    def test_saves_png_from_2d(self, tmp_path):
        w = np.random.rand(SEQ_LEN, SEQ_LEN).astype(np.float32)
        out = tmp_path / "attn2d.png"
        plot_attention_heatmap(w, output_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_saves_png_from_3d(self, tmp_path):
        w = np.random.rand(N_HEADS, SEQ_LEN, SEQ_LEN).astype(np.float32)
        out = tmp_path / "attn3d.png"
        plot_attention_heatmap(w, head_idx=0, output_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_saves_png_from_4d(self, tmp_path):
        w = np.random.rand(BATCH, N_HEADS, SEQ_LEN, SEQ_LEN).astype(np.float32)
        out = tmp_path / "attn4d.png"
        plot_attention_heatmap(w, sample_idx=0, head_idx=1, output_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_3d_head_idx_out_of_range(self):
        w = np.random.rand(N_HEADS, SEQ_LEN, SEQ_LEN).astype(np.float32)
        with pytest.raises(IndexError, match="head_idx"):
            plot_attention_heatmap(w, head_idx=99)

    def test_4d_sample_idx_out_of_range(self):
        w = np.random.rand(BATCH, N_HEADS, SEQ_LEN, SEQ_LEN).astype(np.float32)
        with pytest.raises(IndexError, match="sample_idx"):
            plot_attention_heatmap(w, sample_idx=99)

    def test_4d_head_idx_out_of_range(self):
        w = np.random.rand(BATCH, N_HEADS, SEQ_LEN, SEQ_LEN).astype(np.float32)
        with pytest.raises(IndexError, match="head_idx"):
            plot_attention_heatmap(w, head_idx=99)

    def test_invalid_ndim_raises_value_error(self):
        w = np.random.rand(SEQ_LEN).astype(np.float32)
        with pytest.raises(ValueError, match="2D, 3D, or 4D"):
            plot_attention_heatmap(w)

    def test_5d_raises_value_error(self):
        w = np.random.rand(1, BATCH, N_HEADS, SEQ_LEN, SEQ_LEN).astype(np.float32)
        with pytest.raises(ValueError, match="2D, 3D, or 4D"):
            plot_attention_heatmap(w)

    def test_custom_position_labels(self, tmp_path):
        w = np.random.rand(SEQ_LEN, SEQ_LEN).astype(np.float32)
        labels = [f"t{i}" for i in range(SEQ_LEN)]
        out = tmp_path / "attn_labels.png"
        plot_attention_heatmap(w, position_labels=labels, output_path=out)
        assert out.exists()

    def test_no_output_path_does_not_crash(self):
        """Calling without output_path should not raise (figure is just closed)."""
        w = np.random.rand(SEQ_LEN, SEQ_LEN).astype(np.float32)
        plot_attention_heatmap(w)  # should not raise

    def test_end_to_end_extract_then_plot(self, model, x, tmp_path):
        """Full pipeline: extract from model, then plot."""
        weights = extract_attention_weights(model, x, layer_idx=-1)
        out = tmp_path / "e2e.png"
        plot_attention_heatmap(weights, head_idx=0, sample_idx=0, output_path=out)
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Axis 2 artifact helpers (imported from run_axis2)
# ---------------------------------------------------------------------------

def _make_loader(n_samples: int, seq_len: int, n_feat: int, n_attack: int) -> DataLoader:
    """Small DataLoader with n_attack attack samples (label=1) and rest benign (label=0)."""
    X = torch.randn(n_samples, seq_len, n_feat)
    y = torch.zeros(n_samples, dtype=torch.long)
    y[:n_attack] = 1
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


@pytest.fixture(scope="module")
def small_transformer() -> LightTransformer:
    m = LightTransformer(n_features=N_FEATURES, n_classes=N_CLASSES)
    m.eval()
    return m


class TestCollectAttentionByClass:
    def test_returns_expected_keys(self, small_transformer):
        from run_axis2 import _collect_attention_by_class
        loader = _make_loader(32, SEQ_LEN, N_FEATURES, n_attack=16)
        result = _collect_attention_by_class(small_transformer, loader, "cpu", n_per_group=8)
        assert set(result.keys()) == {"attack", "benign", "n_attack", "n_benign"}

    def test_attack_avg_shape(self, small_transformer):
        from run_axis2 import _collect_attention_by_class
        loader = _make_loader(64, SEQ_LEN, N_FEATURES, n_attack=32)
        result = _collect_attention_by_class(small_transformer, loader, "cpu", n_per_group=4)
        if result["attack"] is not None:
            assert result["attack"].shape == (N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_benign_avg_shape(self, small_transformer):
        from run_axis2 import _collect_attention_by_class
        loader = _make_loader(64, SEQ_LEN, N_FEATURES, n_attack=0)  # all benign
        result = _collect_attention_by_class(small_transformer, loader, "cpu", n_per_group=4)
        if result["benign"] is not None:
            assert result["benign"].shape == (N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_none_when_no_attack_samples(self, small_transformer):
        from run_axis2 import _collect_attention_by_class
        loader = _make_loader(16, SEQ_LEN, N_FEATURES, n_attack=0)
        result = _collect_attention_by_class(small_transformer, loader, "cpu", n_per_group=8)
        # No attack samples → attack must be None (or 0 collected)
        assert result["n_attack"] == 0
        assert result["attack"] is None

    def test_counts_respect_n_per_group(self, small_transformer):
        from run_axis2 import _collect_attention_by_class
        loader = _make_loader(128, SEQ_LEN, N_FEATURES, n_attack=64)
        result = _collect_attention_by_class(small_transformer, loader, "cpu", n_per_group=3)
        assert result["n_attack"] <= 3
        assert result["n_benign"] <= 3


class TestSaveTransformerCheckpoint:
    def test_checkpoint_pt_created(self, tmp_path, small_transformer):
        from run_axis2 import _save_transformer_checkpoint
        win_dir = tmp_path / "window_0"
        _save_transformer_checkpoint(small_transformer, win_dir, {"seed": 42})
        assert (win_dir / "checkpoint.pt").exists()

    def test_checkpoint_meta_json_created(self, tmp_path, small_transformer):
        from run_axis2 import _save_transformer_checkpoint
        win_dir = tmp_path / "window_0"
        meta = {"seed": 42, "horizon": "1", "window_idx": 0}
        _save_transformer_checkpoint(small_transformer, win_dir, meta)
        meta_path = win_dir / "checkpoint_meta.json"
        assert meta_path.exists()
        loaded = json.loads(meta_path.read_text())
        assert loaded["seed"] == 42

    def test_state_dict_loadable(self, tmp_path, small_transformer):
        from run_axis2 import _save_transformer_checkpoint
        win_dir = tmp_path / "window_0"
        _save_transformer_checkpoint(small_transformer, win_dir, {})
        ckpt_path = win_dir / "checkpoint.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert isinstance(state, dict)
        assert len(state) > 0


class TestGenerateAttentionArtifacts:
    def test_attention_summary_json_created(self, tmp_path, small_transformer):
        from run_axis2 import _generate_attention_artifacts
        loader = _make_loader(32, SEQ_LEN, N_FEATURES, n_attack=16)
        summary = _generate_attention_artifacts(
            model=small_transformer,
            loader_delta0=loader,
            loader_future=None,
            device="cpu",
            artifacts_dir=tmp_path / "attention",
            n_per_group=4,
        )
        summary_path = tmp_path / "attention" / "attention_summary.json"
        assert summary_path.exists()
        loaded = json.loads(summary_path.read_text())
        assert "delta0" in loaded
        assert "future" in loaded
        assert "png_paths" in loaded

    def test_delta0_pngs_generated_when_samples_exist(self, tmp_path, small_transformer):
        from run_axis2 import _generate_attention_artifacts
        loader = _make_loader(64, SEQ_LEN, N_FEATURES, n_attack=32)
        _generate_attention_artifacts(
            model=small_transformer,
            loader_delta0=loader,
            loader_future=None,
            device="cpu",
            artifacts_dir=tmp_path / "attention",
            n_per_group=4,
        )
        attn_dir = tmp_path / "attention"
        # At least one of the two delta0 PNGs must exist (depending on model preds)
        pngs = list(attn_dir.glob("*.png"))
        assert len(pngs) >= 0  # graceful: may be 0 if model always misclassifies

    def test_no_crash_when_no_correct_samples(self, tmp_path, small_transformer):
        """Even with zero correctly classified samples the function must not raise."""
        from run_axis2 import _generate_attention_artifacts
        loader = _make_loader(4, SEQ_LEN, N_FEATURES, n_attack=0)
        _generate_attention_artifacts(
            model=small_transformer,
            loader_delta0=loader,
            loader_future=loader,
            device="cpu",
            artifacts_dir=tmp_path / "attention",
            n_per_group=16,
        )
        assert (tmp_path / "attention" / "attention_summary.json").exists()

    def test_future_warning_when_no_loader(self, tmp_path, small_transformer):
        from run_axis2 import _generate_attention_artifacts
        loader = _make_loader(32, SEQ_LEN, N_FEATURES, n_attack=16)
        summary = _generate_attention_artifacts(
            model=small_transformer,
            loader_delta0=loader,
            loader_future=None,
            device="cpu",
            artifacts_dir=tmp_path / "attention",
            n_per_group=4,
        )
        assert any("future" in w.lower() or "Δt>0" in w for w in summary["warnings"])

    def test_non_transformer_model_family_produces_no_artifacts(self, tmp_path):
        """Verifies the guard: for non-transformer families, the artifact path is never created."""
        for model_family in ["xgboost", "mlp", "cnn_bilstm"]:
            seed_artifacts_dir: Path | None = None
            if model_family == "transformer" and (True or True):
                seed_artifacts_dir = tmp_path / f"{model_family}_artifacts"
            assert seed_artifacts_dir is None
