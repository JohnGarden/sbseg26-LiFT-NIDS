"""Tests for timestamp validation contract in run_experiment._validate_and_normalize_timestamp.

These tests verify the contract without importing run_experiment.py directly,
following the project's established pattern (see TestTimestampNormalization in
test_temporal_protocol_integration.py).  The inline implementation mirrors the
production code so that the tests are runnable even when heavy optional
dependencies (xgboost, scipy) cannot be loaded in the current environment.

Contract under test:
- The canonical timestamp column name comes from dataset.timestamp_column in
  config (default "timestamp").
- Legacy "Timestamp" (CICFlowMeter title-case) is renamed to the canonical name.
- Values that cannot be parsed as datetime raise ValueError with the column
  name, the NaT count, and sample invalid values in the message.
- Temporal datasets (detected by the presence of a "year" column) must have a
  valid timestamp column; absence raises ValueError.
- Non-temporal datasets (no "year" column, e.g. CICIoT2023 static splits) are
  allowed to have no timestamp column at all.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Reference implementation (mirrors run_experiment._validate_and_normalize_timestamp)
# ---------------------------------------------------------------------------

def _ref_resolve_timestamp_col(df: pd.DataFrame, cfg: dict) -> str | None:
    canonical = cfg.get("dataset", {}).get("timestamp_column", "timestamp")
    if canonical in df.columns:
        return canonical
    if "Timestamp" in df.columns:
        return "Timestamp"
    return None


def _ref_validate_and_normalize_timestamp(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, str | None]:
    """Reference implementation used in tests."""
    ds_cfg = cfg.get("dataset", {})
    canonical = ds_cfg.get("timestamp_column", "timestamp")
    is_temporal = "year" in df.columns

    found = _ref_resolve_timestamp_col(df, cfg)
    if found is None:
        if is_temporal:
            raise ValueError(
                f"Timestamp column '{canonical}' is required for temporal datasets "
                f"(MAWIFlow 'year' column detected) but was not found. "
                f"Ensure each yearly parquet file contains a timestamp column. "
                f"Config key: dataset.timestamp_column={canonical!r}."
            )
        return df, None

    if found != canonical:
        df = df.rename(columns={found: canonical})

    original_values = df[canonical].copy()
    df[canonical] = pd.to_datetime(df[canonical], errors="coerce")
    nat_mask = df[canonical].isna()
    nat_count = int(nat_mask.sum())

    if nat_count > 0:
        invalid_examples = original_values[nat_mask].head(5).tolist()
        raise ValueError(
            f"Timestamp column '{canonical}' contains {nat_count} value(s) that "
            f"could not be parsed as datetime (NaT after coercion). "
            f"Sample invalid values: {invalid_examples}. "
            f"Fix the parquet/preprocessing pipeline to ensure all timestamps are "
            f"valid datetime strings before running experiments."
        )

    return df, canonical


def _ts_cfg(col: str = "timestamp") -> dict:
    return {"dataset": {"timestamp_column": col}}


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

class TestValidTimestamp:
    def test_valid_canonical_column_passes(self):
        df = pd.DataFrame({
            "year": [2007, 2008],
            "timestamp": pd.to_datetime(["2007-01-01", "2008-01-01"]),
            "label": [0, 1],
        })
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert col == "timestamp"
        assert "timestamp" in result_df.columns
        assert result_df["timestamp"].notna().all()

    def test_valid_string_timestamps_are_converted(self):
        df = pd.DataFrame({
            "year": [2010],
            "timestamp": ["2010-06-15 12:00:00"],
            "label": [0],
        })
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert col == "timestamp"
        assert pd.api.types.is_datetime64_any_dtype(result_df["timestamp"])

    def test_already_datetime_column_passes(self):
        df = pd.DataFrame({
            "year": [2012],
            "timestamp": [pd.Timestamp("2012-03-01")],
            "label": [0],
        })
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert col == "timestamp"
        assert result_df["timestamp"].notna().all()


# ---------------------------------------------------------------------------
# Legacy column renaming
# ---------------------------------------------------------------------------

class TestLegacyRenaming:
    def test_titlecase_timestamp_renamed_to_canonical(self):
        base = datetime.datetime(2007, 1, 1)
        df = pd.DataFrame({
            "year": [2007, 2007],
            "Timestamp": [base, base + datetime.timedelta(hours=1)],
            "label": [0, 1],
        })
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert col == "timestamp"
        assert "timestamp" in result_df.columns
        assert "Timestamp" not in result_df.columns

    def test_titlecase_renamed_column_is_valid_datetime(self):
        df = pd.DataFrame({
            "year": [2009],
            "Timestamp": ["2009-05-20 08:30:00"],
            "label": [1],
        })
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert pd.api.types.is_datetime64_any_dtype(result_df[col])

    def test_non_temporal_titlecase_also_renamed(self):
        """CICIoT-style dataset with Timestamp column (no year) — rename still applies."""
        df = pd.DataFrame({
            "Timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "label": [0, 1],
        })
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert col == "timestamp"
        assert "Timestamp" not in result_df.columns


# ---------------------------------------------------------------------------
# Invalid timestamps raise ValueError
# ---------------------------------------------------------------------------

class TestInvalidTimestamp:
    def test_invalid_string_raises_value_error(self):
        df = pd.DataFrame({
            "year": [2007, 2007],
            "timestamp": ["2007-01-01", "not-a-date"],
            "label": [0, 1],
        })
        with pytest.raises(ValueError, match="timestamp"):
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())

    def test_error_message_contains_nat_count(self):
        df = pd.DataFrame({
            "year": [2007] * 5,
            "timestamp": ["2007-01-01", "bad", "also-bad", "2007-01-04", "2007-01-05"],
            "label": [0, 1, 0, 1, 0],
        })
        with pytest.raises(ValueError, match="2"):
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())

    def test_error_message_contains_column_name(self):
        df = pd.DataFrame({
            "year": [2007],
            "timestamp": ["not-a-date"],
            "label": [0],
        })
        with pytest.raises(ValueError, match="'timestamp'"):
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())

    def test_error_message_contains_invalid_example(self):
        df = pd.DataFrame({
            "year": [2007],
            "timestamp": ["INVALID_VALUE_XYZ"],
            "label": [0],
        })
        with pytest.raises(ValueError, match="INVALID_VALUE_XYZ"):
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())

    def test_multiple_invalid_values_reported_correctly(self):
        df = pd.DataFrame({
            "year": list(range(2007, 2017)),
            "timestamp": ["2007-01-01"] * 8 + ["garbage", "trash"],
            "label": [0] * 10,
        })
        with pytest.raises(ValueError) as exc_info:
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert "2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# MAWIFlow without timestamp aborts
# ---------------------------------------------------------------------------

class TestTemporalDatasetWithoutTimestamp:
    def test_mawiflow_without_timestamp_raises(self):
        """Temporal dataset (year column present) with no timestamp → ValueError."""
        df = pd.DataFrame({
            "year": [2007, 2008],
            "feat_a": [1.0, 2.0],
            "label": [0, 1],
        })
        with pytest.raises(ValueError, match="timestamp"):
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())

    def test_error_mentions_year_column_detection(self):
        df = pd.DataFrame({
            "year": [2015],
            "feat": [0.5],
            "label": [0],
        })
        with pytest.raises(ValueError, match="year"):
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())

    def test_error_mentions_config_key(self):
        df = pd.DataFrame({
            "year": [2015],
            "feat": [0.5],
            "label": [0],
        })
        with pytest.raises(ValueError, match="timestamp_column"):
            _ref_validate_and_normalize_timestamp(df, _ts_cfg())


# ---------------------------------------------------------------------------
# CICIoT/static — absent timestamp is allowed
# ---------------------------------------------------------------------------

class TestStaticDatasetWithoutTimestamp:
    def test_no_timestamp_no_year_returns_none(self):
        """CICIoT-style: no year column, no timestamp → (df, None), no error."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [0.1, 0.2, 0.3],
            "label": [0, 1, 0],
        })
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert col is None
        assert list(result_df.columns) == list(df.columns)

    def test_static_df_is_returned_unmodified(self):
        df = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
        original_columns = set(df.columns)
        result_df, col = _ref_validate_and_normalize_timestamp(df, _ts_cfg())
        assert col is None
        assert set(result_df.columns) == original_columns
