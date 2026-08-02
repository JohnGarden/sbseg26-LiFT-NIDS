"""Verify CICIoT2023 binary-classification contract.

Guarantees:
- label_multiclass is excluded from the feature set by _LABEL_COLS.
- label (the binary target) is also excluded from features.
- get_feature_cols returns only non-target, non-metadata columns.
- The pipeline constant n_classes=2 is consistent with the binary label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts/ to path so run_experiment can be imported in tests.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_experiment import _LABEL_COLS, get_feature_cols  # noqa: E402


def _make_ciciot_like_df(n: int = 20) -> pd.DataFrame:
    """Minimal DataFrame mimicking CICIoT2023 column layout."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feat_flow_duration": rng.random(n).astype("float32"),
            "feat_pkt_len_mean": rng.random(n).astype("float32"),
            "label": rng.integers(0, 2, n).astype("int64"),
            "label_multiclass": rng.integers(0, 8, n).astype("int64"),
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="s"),
            "flow_id": [f"flow_{i}" for i in range(n)],
        }
    )


class TestCICIoT2023BinaryContract:
    def test_label_multiclass_in_label_cols_constant(self) -> None:
        assert "label_multiclass" in _LABEL_COLS

    def test_label_in_label_cols_constant(self) -> None:
        assert "label" in _LABEL_COLS

    def test_label_multiclass_excluded_from_feature_cols(self) -> None:
        df = _make_ciciot_like_df()
        num_cols, cat_cols = get_feature_cols(df, label_col="label")
        all_features = set(num_cols + cat_cols)
        assert "label_multiclass" not in all_features, (
            "label_multiclass must not appear as a feature — it is audit-only metadata"
        )

    def test_label_excluded_from_feature_cols(self) -> None:
        df = _make_ciciot_like_df()
        num_cols, cat_cols = get_feature_cols(df, label_col="label")
        all_features = set(num_cols + cat_cols)
        assert "label" not in all_features

    def test_feature_cols_contains_only_feature_columns(self) -> None:
        df = _make_ciciot_like_df()
        num_cols, cat_cols = get_feature_cols(df, label_col="label")
        all_features = set(num_cols + cat_cols)
        assert "feat_flow_duration" in all_features
        assert "feat_pkt_len_mean" in all_features

    def test_timestamp_and_flow_id_excluded(self) -> None:
        df = _make_ciciot_like_df()
        num_cols, cat_cols = get_feature_cols(df, label_col="label")
        all_features = set(num_cols + cat_cols)
        assert "timestamp" not in all_features
        assert "flow_id" not in all_features

    def test_binary_label_values_are_zero_and_one(self) -> None:
        df = _make_ciciot_like_df(100)
        unique = set(df["label"].unique())
        assert unique <= {0, 1}, f"Binary label must be {{0, 1}}, got {unique}"

    def test_n_classes_is_two(self) -> None:
        """The pipeline constant passed to all CICIoT experiments is n_classes=2."""
        from lift_nids.data.dataset import FlowDataset

        rng = np.random.default_rng(0)
        labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        features = rng.random((5, 4)).astype("float32")
        ds = FlowDataset(features, labels)
        assert ds.n_classes == 2
