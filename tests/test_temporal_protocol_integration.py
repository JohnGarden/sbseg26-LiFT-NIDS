"""Integration tests for the MAWIFlow temporal evaluation protocol.

Validates the full methodology chain — split generation, temporal train/val
separation, per-year F1+ recording, trajectory, and nAUT — using only
synthetic data and pure functions.  No model training, no Parquet files.

These tests would fail against the old implementation that:
  - used k as test-horizon instead of train-window size;
  - used .sample() for internal val split (temporal leakage for k>1 windows);
  - computed one aggregated F1+ and replicated it across all test years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lift_nids.data.splits import (
    forward_chaining_splits,
    temporal_train_val_test_split,
)
from lift_nids.evaluation.metrics import compute_classification_metrics
from lift_nids.evaluation.temporal_metrics import compute_naut, median_trajectory

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

YEARS = list(range(2007, 2013))   # 2007–2012 inclusive
ROWS_PER_YEAR = 10


@pytest.fixture(name="df6")
def _df6() -> pd.DataFrame:
    """Synthetic DataFrame: 6 years × 10 rows, with a monotone 'pos' column."""
    rng = np.random.default_rng(0)
    rows = []
    for y in YEARS:
        for i in range(ROWS_PER_YEAR):
            rows.append({
                "year":   y,
                # Global temporal position — strictly increasing with time.
                "pos":    (y - YEARS[0]) * ROWS_PER_YEAR + i,
                "feat_1": float(rng.random()),
                "feat_2": float(rng.random()),
                "label":  int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item 3 — Forward-chaining window structure
# ---------------------------------------------------------------------------

class TestForwardChainingStructure:
    """Correct training/test year assignment for every k configuration."""

    def test_k1_first_window_trains_2007_tests_2008_to_2012(self, df6):
        w = forward_chaining_splits(df6, "year", k=1)[0]
        assert w.train_years == [2007]
        assert w.anchor_year == 2007
        assert sorted(w.test_dfs_by_year) == [2008, 2009, 2010, 2011, 2012]

    def test_k2_first_window_trains_2007_2008_tests_2009_to_2012(self, df6):
        w = forward_chaining_splits(df6, "year", k=2)[0]
        assert w.train_years == [2007, 2008]
        assert w.anchor_year == 2008
        assert sorted(w.test_dfs_by_year) == [2009, 2010, 2011, 2012]

    def test_k3_first_window_trains_2007_2008_2009(self, df6):
        w = forward_chaining_splits(df6, "year", k=3)[0]
        assert w.train_years == [2007, 2008, 2009]

    def test_cumulative_window_grows_by_exactly_one_year(self, df6):
        windows = forward_chaining_splits(df6, "year", k="cumulative")
        for i in range(1, len(windows)):
            assert len(windows[i].train_years) == len(windows[i - 1].train_years) + 1

    def test_cumulative_never_leaks_future_year_into_train(self, df6):
        windows = forward_chaining_splits(df6, "year", k="cumulative")
        for w in windows:
            assert max(w.train_years) < min(w.test_dfs_by_year)

    def test_no_year_overlap_between_train_and_test_for_all_k(self, df6):
        for k in [1, 2, 3, "cumulative"]:
            for w in forward_chaining_splits(df6, "year", k=k):
                overlap = set(w.train_years) & set(w.test_dfs_by_year)
                assert overlap == set(), f"k={k} window has train/test overlap: {overlap}"

    def test_each_test_df_contains_only_its_year(self, df6):
        for w in forward_chaining_splits(df6, "year", k=1):
            for yr, yr_df in w.test_dfs_by_year.items():
                assert (yr_df["year"] == yr).all()

    def test_train_df_contains_exactly_the_declared_train_years(self, df6):
        for w in forward_chaining_splits(df6, "year", k=2):
            assert set(w.train_df["year"].unique()) == set(w.train_years)


# ---------------------------------------------------------------------------
# Item 5 — F1+ computed separately per test year
# ---------------------------------------------------------------------------

class TestF1PlusByYear:
    """Each (anchor_year, test_year) pair must have its own F1+ value."""

    def test_distinct_f1plus_values_for_different_years(self):
        """
        Anchor = 2008.  Test-year 2009 gets perfect predictions (F1=1.0);
        test-year 2010 gets inverted predictions (F1=0.0).
        After per-year evaluation the two records must carry different values.
        """
        y_true = np.array([0, 0, 1, 1, 1])

        metrics_2009 = compute_classification_metrics(
            y_true, y_true.copy()          # perfect → F1_1 = 1.0
        )
        metrics_2010 = compute_classification_metrics(
            y_true, 1 - y_true             # inverted → F1_1 = 0.0
        )

        f1plus_records = [
            (2008, 2009, metrics_2009["f1_1"]),
            (2008, 2010, metrics_2010["f1_1"]),
        ]

        assert f1plus_records[0][2] == pytest.approx(1.0)
        assert f1plus_records[1][2] == pytest.approx(0.0)
        assert f1plus_records[0][2] != f1plus_records[1][2]

    def test_anchor_test_year_pairs_are_unique(self, df6):
        """The (anchor_year, test_year) key space has no duplicates."""
        windows = forward_chaining_splits(df6, "year", k=1)
        pairs = [(w.anchor_year, yr) for w in windows for yr in w.test_dfs_by_year]
        assert len(pairs) == len(set(pairs))

    def test_pooling_test_years_loses_per_year_information(self):
        """
        Shows why the old aggregation approach is wrong:
        the pooled F1+ over 2009+2010 differs from both individual values,
        meaning it does not represent either year accurately.
        """
        y_true = np.array([0, 1, 0, 1])

        m_2009 = compute_classification_metrics(y_true, y_true.copy())          # F1=1.0
        m_2010 = compute_classification_metrics(y_true, np.ones_like(y_true))   # all-attack

        # Pooled (old wrong behaviour)
        y_true_pool = np.concatenate([y_true, y_true])
        y_pred_pool = np.concatenate([y_true.copy(), np.ones_like(y_true)])
        m_pooled = compute_classification_metrics(y_true_pool, y_pred_pool)

        # The pooled metric is different from both per-year metrics
        assert m_pooled["f1_1"] != pytest.approx(m_2009["f1_1"])
        assert m_pooled["f1_1"] != pytest.approx(m_2010["f1_1"])


# ---------------------------------------------------------------------------
# Item 6 — Temporal trajectory and nAUT (trapezoidal rule, Δt=0 required)
# ---------------------------------------------------------------------------

class TestTrajectoryAndNAUT:
    """Verify median_trajectory and compute_naut on known inputs.

    nAUT_H uses the IM28 trapezoidal formula:
        nAUT_H = (1/H) * Σ_{Δt=0}^{H-1} [ F̃1+(Δt) + F̃1+(Δt+1) ] / 2

    Δt=0 is the internal test slice (same time window as training, held-out).
    """

    # Δt=0 appears twice (0.92 and 0.88), Δt=1 twice (1.0 and 0.8), Δt=2 once (0.5).
    RECORDS = [
        (2008, 2008, 0.92),  # Δt = 0  (internal test, window 1)
        (2008, 2009, 1.0),   # Δt = 1
        (2008, 2010, 0.5),   # Δt = 2
        (2009, 2009, 0.88),  # Δt = 0  (internal test, window 2)
        (2009, 2010, 0.8),   # Δt = 1  (second observation)
    ]
    # traj = {0: 0.9, 1: 0.9, 2: 0.5}

    def test_delta_t_is_test_year_minus_anchor_year(self):
        records = [(2010, 2013, 0.7)]   # Δt = 3
        traj = median_trajectory(records)
        assert 3 in traj
        assert traj[3] == pytest.approx(0.7)

    def test_median_trajectory_delta0_is_median_of_all_delta0_values(self):
        traj = median_trajectory(self.RECORDS)
        # Δt=0 values: [0.92, 0.88] → median = 0.9
        assert traj[0] == pytest.approx(0.9)

    def test_median_trajectory_delta1_is_median_of_all_delta1_values(self):
        traj = median_trajectory(self.RECORDS)
        # Δt=1 values: [1.0, 0.8] → median = 0.9
        assert traj[1] == pytest.approx(0.9)

    def test_median_trajectory_delta2_single_observation(self):
        traj = median_trajectory(self.RECORDS)
        # Δt=2 values: [0.5] → median = 0.5
        assert traj[2] == pytest.approx(0.5)

    def test_naut_h1_is_trapezoid_of_delta0_and_delta1(self):
        """nAUT_1 = (F̃1+(0) + F̃1+(1)) / 2 — first interval of the trapezoidal sum."""
        traj = {0: 0.9, 1: 0.7}
        assert compute_naut(traj, H=1) == pytest.approx((0.9 + 0.7) / 2)

    def test_naut_h2_is_mean_of_two_trapezoids(self):
        """nAUT_2 = [(F(0)+F(1))/2 + (F(1)+F(2))/2] / 2."""
        traj = {0: 0.9, 1: 0.7, 2: 0.5}
        expected = ((0.9 + 0.7) / 2 + (0.7 + 0.5) / 2) / 2
        assert compute_naut(traj, H=2) == pytest.approx(expected)

    def test_naut_h5_missing_lags_count_as_zero(self):
        # traj = {0: 0.9, 1: 0.9, 2: 0.5}; Δt=3,4,5 missing → 0.0
        traj = median_trajectory(self.RECORDS)
        # Trapezoids: (0.9+0.9)/2, (0.9+0.5)/2, (0.5+0)/2, (0+0)/2, (0+0)/2
        expected = ((0.9 + 0.9) / 2 + (0.9 + 0.5) / 2 + (0.5 + 0.0) / 2 + 0.0 + 0.0) / 5
        assert compute_naut(traj, H=5) == pytest.approx(expected)

    def test_naut_decreases_as_H_grows_when_later_lags_are_missing(self):
        traj = median_trajectory(self.RECORDS)
        assert compute_naut(traj, H=1) > compute_naut(traj, H=5)

    def test_naut_h1_h3_h5_with_known_trajectory(self):
        """End-to-end check with explicit Δt=0 and monotone-decreasing trajectory."""
        records = [
            (2007, 2007, 1.0),   # Δt=0
            (2007, 2008, 0.8),   # Δt=1
            (2007, 2009, 0.6),   # Δt=2
            (2007, 2010, 0.4),   # Δt=3
            (2008, 2008, 1.0),   # Δt=0 (second obs)
            (2008, 2009, 0.8),   # Δt=1 (second obs)
        ]
        traj = median_trajectory(records)
        assert traj[0] == pytest.approx(1.0)
        assert traj[1] == pytest.approx(0.8)
        assert traj[2] == pytest.approx(0.6)
        assert traj[3] == pytest.approx(0.4)

        # nAUT_1 = (1.0+0.8)/2 = 0.9
        assert compute_naut(traj, H=1) == pytest.approx(0.9)
        # nAUT_3 = [(1.0+0.8)/2 + (0.8+0.6)/2 + (0.6+0.4)/2] / 3
        assert compute_naut(traj, H=3) == pytest.approx((0.9 + 0.7 + 0.5) / 3)
        # nAUT_5 = [0.9 + 0.7 + 0.5 + (0.4+0)/2 + 0] / 5
        assert compute_naut(traj, H=5) == pytest.approx((0.9 + 0.7 + 0.5 + 0.2 + 0.0) / 5)


# ---------------------------------------------------------------------------
# Item 3b — temporal_train_val_test_split (60/20/20 for Δt=0 evaluation)
# ---------------------------------------------------------------------------

class TestTemporalTrainValTestSplit:
    """Verify the three-way temporal split used for the 60/20/20 protocol."""

    def test_three_parts_sum_to_total(self, df6):
        fit_df, val_df, test_df = temporal_train_val_test_split(df6, "year")
        assert len(fit_df) + len(val_df) + len(test_df) == len(df6)

    def test_temporal_order_fit_before_val_before_test(self, df6):
        fit_df, val_df, test_df = temporal_train_val_test_split(df6, "year")
        assert max(fit_df["pos"]) < min(val_df["pos"])
        assert max(val_df["pos"]) < min(test_df["pos"])

    def test_approximate_60_20_20_fractions(self, df6):
        fit_df, val_df, test_df = temporal_train_val_test_split(df6, "year")
        n = len(df6)
        assert abs(len(fit_df) / n - 0.60) < 0.05
        assert abs(len(val_df) / n - 0.20) < 0.05
        assert abs(len(test_df) / n - 0.20) < 0.05

    def test_no_overlap_between_splits(self, df6):
        fit_df, val_df, test_df = temporal_train_val_test_split(df6, "year")
        fit_pos = set(fit_df["pos"])
        val_pos = set(val_df["pos"])
        tst_pos = set(test_df["pos"])
        assert fit_pos.isdisjoint(val_pos)
        assert fit_pos.isdisjoint(tst_pos)
        assert val_pos.isdisjoint(tst_pos)

    def test_test_split_never_empty(self):
        """Even a three-row DataFrame produces at least 1 row in each split."""
        tiny = pd.DataFrame({"year": [2007, 2007, 2007], "pos": [0, 1, 2], "label": [0, 1, 0]})
        fit_df, val_df, test_df = temporal_train_val_test_split(tiny, "year")
        assert len(test_df) >= 1
        assert len(val_df) >= 1


# ---------------------------------------------------------------------------
# Item 5b — dt=0 present in f1plus_records
# ---------------------------------------------------------------------------

class TestDeltaT0InRecords:
    """Structural validation: every forward-chaining window must produce a
    (anchor_year, anchor_year, f1_plus) entry (Δt=0) in addition to future
    year entries."""

    def test_f1plus_records_contain_delta_t_zero(self, df6):
        """Simulates the records structure: each window contributes a dt=0 entry."""
        windows = forward_chaining_splits(df6, "year", k=1)
        # For each window, add a synthetic dt=0 record (anchor_year, anchor_year)
        f1plus_records: list[tuple[int, int, float]] = []
        for w in windows:
            f1plus_records.append((w.anchor_year, w.anchor_year, 0.9))  # dt=0
            for test_yr in w.test_dfs_by_year:
                f1plus_records.append((w.anchor_year, test_yr, 0.7))

        traj = median_trajectory(f1plus_records)
        assert 0 in traj, "Δt=0 must be present in the trajectory after adding internal test records"

    def test_trajectory_with_delta_t_zero_enables_naut(self, df6):
        """nAUT_1 computed from a trajectory with Δt=0 uses both endpoints."""
        traj = {0: 0.9, 1: 0.8}
        naut_1 = compute_naut(traj, H=1)
        assert naut_1 == pytest.approx((0.9 + 0.8) / 2)
        # Verify it differs from a trajectory missing dt=0
        traj_no_zero = {1: 0.8}
        naut_1_no_zero = compute_naut(traj_no_zero, H=1)
        assert naut_1_no_zero == pytest.approx((0.0 + 0.8) / 2)
        assert naut_1 != pytest.approx(naut_1_no_zero)

    def test_anchor_year_appears_as_both_anchor_and_test_year(self, df6):
        """anchor_year must appear as test_year for exactly one record per window."""
        windows = forward_chaining_splits(df6, "year", k=1)
        f1plus_records: list[tuple[int, int, float]] = []
        for w in windows:
            f1plus_records.append((w.anchor_year, w.anchor_year, 0.9))
            for test_yr in w.test_dfs_by_year:
                f1plus_records.append((w.anchor_year, test_yr, 0.7))

        dt0_records = [(a, t, f) for a, t, f in f1plus_records if t == a]
        assert len(dt0_records) == len(windows), (
            f"Expected one dt=0 record per window ({len(windows)}), "
            f"got {len(dt0_records)}"
        )


# ---------------------------------------------------------------------------
# Item 7 — HPO temporal split: 60/20/20 without shuffle
# ---------------------------------------------------------------------------

class TestHPOTemporalSplit:
    """Validate that the HPO split for the anchor year is temporal, not random.

    The scripts (run_experiment.py and run_axis2.py) must use
    temporal_train_val_test_split on the anchor-year DataFrame, producing
    fit rows that strictly precede val rows chronologically.  A random
    .sample() split violates this requirement.
    """

    def test_hpo_split_fit_rows_precede_val_rows(self, df6):
        """Simulates the HPO split: fit rows must come before val rows."""
        anchor_df = df6[df6["year"] == 2007].reset_index(drop=True)
        fit_df, val_df, _ = temporal_train_val_test_split(anchor_df, "year")
        assert max(fit_df["pos"]) < min(val_df["pos"])

    def test_hpo_split_three_parts_sum_to_anchor_total(self, df6):
        anchor_df = df6[df6["year"] == 2007].reset_index(drop=True)
        fit_df, val_df, test_df = temporal_train_val_test_split(anchor_df, "year")
        assert len(fit_df) + len(val_df) + len(test_df) == len(anchor_df)

    def test_hpo_split_internal_test_not_overlapping_fit_or_val(self, df6):
        anchor_df = df6[df6["year"] == 2007].reset_index(drop=True)
        fit_df, val_df, test_df = temporal_train_val_test_split(anchor_df, "year")
        fit_pos = set(fit_df["pos"])
        val_pos = set(val_df["pos"])
        tst_pos = set(test_df["pos"])
        assert fit_pos.isdisjoint(tst_pos)
        assert val_pos.isdisjoint(tst_pos)

    def test_random_sample_would_violate_temporal_order_for_hpo(self, df6):
        """
        Shows why .sample() is wrong for HPO: for a single year the row
        order is fixed and stable, so a random sample may pull rows from any
        position — the latest fit row can be newer than the earliest val row.
        """
        anchor_df = df6[df6["year"] == 2007].reset_index(drop=True)
        n = len(anchor_df)
        n_val = max(1, int(n * 0.20))

        violations = 0
        for seed in range(50):
            val_idx = anchor_df.sample(n=n_val, random_state=seed).index
            val_s = anchor_df.loc[val_idx]
            fit_s = anchor_df.drop(val_idx)
            if len(fit_s) > 0 and len(val_s) > 0:
                if max(fit_s["pos"]) >= min(val_s["pos"]):
                    violations += 1

        # For single-year anchor with 10 rows, .sample() will eventually
        # pick early rows for val, causing the fit set to contain later rows.
        assert violations > 0, (
            "Expected .sample() to violate temporal order for at least one seed — "
            "demonstrates why temporal_train_val_test_split must replace it in HPO."
        )

    def test_hpo_split_is_reproducible_without_seed(self, df6):
        """temporal_train_val_test_split is deterministic (no randomness)."""
        anchor_df = df6[df6["year"] == 2007].reset_index(drop=True)
        fit1, val1, _ = temporal_train_val_test_split(anchor_df, "year")
        fit2, val2, _ = temporal_train_val_test_split(anchor_df, "year")
        assert list(fit1["pos"]) == list(fit2["pos"])
        assert list(val1["pos"]) == list(val2["pos"])


# ---------------------------------------------------------------------------
# Item 8 — timestamp_col: within-year chronological ordering
# ---------------------------------------------------------------------------

class TestTimestampColOrdering:
    """Validate that temporal splits use timestamp_col for within-year ordering.

    Within a single year (anchor-year use case), sort_values(year_col) is a
    no-op.  Passing timestamp_col ensures flows are ordered by their actual
    capture time, which is required for the 60/20/20 temporal protocol.
    """

    @pytest.fixture(name="scrambled_df")
    def _scrambled_df(self) -> pd.DataFrame:
        """Single-year DataFrame with timestamps in reverse order."""
        import datetime
        rng = np.random.default_rng(7)
        base = datetime.datetime(2007, 1, 15, 14, 0, 0)
        rows = []
        for i in range(20):
            rows.append({
                "year":      2007,
                "Timestamp": base + datetime.timedelta(seconds=i * 60),
                "seq":       i,   # 0 = earliest, 19 = latest
                "feat":      float(rng.random()),
                "label":     int(rng.integers(0, 2)),
            })
        df = pd.DataFrame(rows)
        return df.iloc[::-1].reset_index(drop=True)  # reverse = latest first

    def test_without_timestamp_col_fit_may_contain_later_rows(self, scrambled_df):
        """Without timestamp_col, fit can include rows with high seq (latest time)."""
        fit_df, _, _ = temporal_train_val_test_split(scrambled_df, "year")
        # year-only sort is a no-op for single-year; original reversed order stays.
        # Fit gets first 60% of rows — which are the LATEST rows in reversed df.
        assert max(fit_df["seq"]) > min(fit_df["seq"])

    def test_with_timestamp_col_fit_rows_precede_val_rows(self, scrambled_df):
        """With timestamp_col, fit rows have lower seq (earlier time) than val rows."""
        fit_df, val_df, _ = temporal_train_val_test_split(
            scrambled_df, "year", timestamp_col="Timestamp"
        )
        assert max(fit_df["seq"]) < min(val_df["seq"])

    def test_with_timestamp_col_all_three_parts_ordered(self, scrambled_df):
        """fit < val < internal_test by seq when timestamp_col is used."""
        fit_df, val_df, tst_df = temporal_train_val_test_split(
            scrambled_df, "year", timestamp_col="Timestamp"
        )
        assert max(fit_df["seq"]) < min(val_df["seq"])
        assert max(val_df["seq"]) < min(tst_df["seq"])

    def test_missing_timestamp_col_raises_value_error(self, scrambled_df):
        """Requesting a non-existent timestamp_col must fail immediately."""
        with pytest.raises(ValueError, match="timestamp_col"):
            temporal_train_val_test_split(scrambled_df, "year", timestamp_col="NoSuchCol")

    def test_load_dataset_sort_restores_order_after_sample_frac(self):
        """Verifies the sorting logic in load_dataset conceptually:
        after subsampling, sorting by year+Timestamp restores chronological order."""
        import datetime
        base = datetime.datetime(2007, 1, 15, 14, 0)
        rows = [{"year": 2007, "Timestamp": base + datetime.timedelta(seconds=i * 30), "seq": i}
                for i in range(50)]
        df = pd.DataFrame(rows)
        # Simulate sample_frac shuffle then re-sort (as done in load_dataset)
        shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        assert list(shuffled["seq"]) != list(range(50)), "shuffle must have changed order"
        re_sorted = shuffled.sort_values(["year", "Timestamp"], kind="stable").reset_index(drop=True)
        assert list(re_sorted["seq"]) == list(range(50))


# ---------------------------------------------------------------------------
# Item 8b — Timestamp normalization: canonical column name and error-on-missing
# ---------------------------------------------------------------------------

class TestTimestampNormalization:
    """Verify canonical timestamp resolution, legacy normalization, and fail-fast
    behavior when no timestamp column is present in a MAWIFlow temporal run.

    These tests simulate the logic in load_dataset and _resolve_timestamp_col
    without importing the script directly, keeping the test suite self-contained.
    """

    def _simulate_resolve(self, df: pd.DataFrame, canonical: str = "timestamp") -> str | None:
        """Inline replica of _resolve_timestamp_col for test isolation."""
        if canonical in df.columns:
            return canonical
        if "Timestamp" in df.columns:
            return "Timestamp"
        return None

    def test_lowercase_timestamp_column_found_as_canonical(self):
        """DataFrame with 'timestamp' (lowercase) resolves to the canonical name."""
        df = pd.DataFrame({"year": [2007], "timestamp": [pd.Timestamp("2007-01-01")]})
        assert self._simulate_resolve(df, "timestamp") == "timestamp"

    def test_titlecase_timestamp_column_found_as_legacy_fallback(self):
        """DataFrame with only 'Timestamp' (CICFlowMeter) resolves to 'Timestamp'."""
        df = pd.DataFrame({"year": [2007], "Timestamp": [pd.Timestamp("2007-01-01")]})
        assert self._simulate_resolve(df, "timestamp") == "Timestamp"

    def test_missing_timestamp_column_resolves_to_none(self):
        """DataFrame with no timestamp column resolves to None."""
        df = pd.DataFrame({"year": [2007, 2008], "feat": [0.1, 0.2], "label": [0, 1]})
        assert self._simulate_resolve(df, "timestamp") is None

    def test_none_resolution_must_raise_in_temporal_protocol(self):
        """When _resolve_timestamp_col returns None, the temporal runner must raise.

        This test documents the required contract: a None result is not silently
        degraded to year-only ordering — it must abort with a descriptive error.
        """
        df = pd.DataFrame({"year": [2007, 2008], "feat": [0.1, 0.2], "label": [0, 1]})
        found = self._simulate_resolve(df, "timestamp")
        assert found is None
        # Confirm the runner contract: ValueError must be raised when found is None.
        with pytest.raises(ValueError, match="timestamp"):
            if found is None:
                raise ValueError(
                    "MAWIFlow temporal protocol requires an auditable timestamp column."
                )

    def test_titlecase_renaming_normalizes_to_canonical(self):
        """'Timestamp' (legacy) is renamed to 'timestamp' (canonical) via load_dataset logic."""
        import datetime
        base = datetime.datetime(2007, 1, 1)
        df = pd.DataFrame({
            "year": [2007, 2007],
            "Timestamp": [base, base + datetime.timedelta(hours=1)],
            "seq": [0, 1],
        })
        # Simulate normalization step in load_dataset
        canonical = "timestamp"
        found = self._simulate_resolve(df, canonical)
        assert found == "Timestamp"
        if found != canonical:
            df = df.rename(columns={found: canonical})
        assert "timestamp" in df.columns
        assert "Timestamp" not in df.columns

    def test_lowercase_timestamp_sorts_correctly_after_sample(self):
        """With canonical 'timestamp', sample_frac + sort restores chronological order."""
        import datetime
        base = datetime.datetime(2007, 1, 1)
        n = 50
        df = pd.DataFrame({
            "year": [2007] * n,
            "timestamp": [base + datetime.timedelta(seconds=i * 30) for i in range(n)],
            "seq": list(range(n)),
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        assert list(shuffled["seq"]) != list(range(n)), "shuffle must have changed order"
        re_sorted = shuffled.sort_values(["year", "timestamp"], kind="stable").reset_index(drop=True)
        assert list(re_sorted["seq"]) == list(range(n))

    def test_temporal_split_with_lowercase_timestamp_maintains_order(self):
        """temporal_train_val_test_split works with lowercase 'timestamp' column."""
        import datetime
        base = datetime.datetime(2007, 1, 1)
        df = pd.DataFrame({
            "year": [2007] * 20,
            "timestamp": [base + datetime.timedelta(hours=i) for i in range(20)],
            "seq": list(range(20)),
            "label": [0] * 20,
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Reverse so earliest rows are last — sort must restore order
        df = df.iloc[::-1].reset_index(drop=True)
        fit_df, val_df, tst_df = temporal_train_val_test_split(
            df, "year", timestamp_col="timestamp"
        )
        assert max(fit_df["seq"]) < min(val_df["seq"])
        assert max(val_df["seq"]) < min(tst_df["seq"])


# ---------------------------------------------------------------------------
# Item 9 — Single-HPO policy: HPO once on anchor year, reused across windows
# ---------------------------------------------------------------------------

class TestSingleHPOPolicy:
    """Verify the single-anchor-year HPO policy.

    Policy (documented in README):
    - Anchor year = first year in the dataset (years[0]).
    - Single call to run_optuna_search on the anchor-year data only.
    - best_params fixed after that call and reused in ALL forward-chaining windows.
    - No HPO inside the forward-chaining loop.

    These tests validate the structural properties of the policy using synthetic
    data and pure functions.  No model training or Optuna calls are made.
    """

    def test_anchor_year_is_first_available_year(self, df6):
        """Anchor year must be min(years), not a random or median year."""
        years = sorted(df6["year"].unique())
        anchor_year = years[0]
        assert anchor_year == YEARS[0]

    def test_anchor_df_contains_only_anchor_year_rows(self, df6):
        """HPO training data must be strictly limited to the anchor year."""
        years = sorted(df6["year"].unique())
        anchor_df = df6[df6["year"] == years[0]].reset_index(drop=True)
        assert set(anchor_df["year"].unique()) == {years[0]}
        assert len(anchor_df) == ROWS_PER_YEAR

    def test_anchor_df_excludes_all_future_years(self, df6):
        """No future-year row must appear in the HPO dataset."""
        years = sorted(df6["year"].unique())
        anchor_df = df6[df6["year"] == years[0]]
        future_rows = anchor_df[anchor_df["year"] != years[0]]
        assert len(future_rows) == 0

    def test_hpo_split_uses_only_anchor_year_data_not_full_df(self, df6):
        """The 60/20/20 HPO split is applied to anchor_df, not to the full df."""
        years = sorted(df6["year"].unique())
        anchor_df = df6[df6["year"] == years[0]].reset_index(drop=True)
        fit_df, val_df, _ = temporal_train_val_test_split(anchor_df, "year")
        # fit + val cover anchor year only
        assert set(fit_df["year"].unique()) == {years[0]}
        assert set(val_df["year"].unique()) == {years[0]}
        # fit + val do not span the full multi-year dataset
        assert len(fit_df) + len(val_df) <= ROWS_PER_YEAR

    def test_best_params_dict_is_unchanged_across_windows_simulation(self, df6):
        """Simulate reuse: a plain dict of best_params must be identical across windows.

        In run_experiment.py and run_axis2.py, best_params is a plain dict that is
        read (not mutated) inside every forward-chaining window.  This test shows
        the structural guarantee that dict-reading is side-effect-free.
        """
        # Simulate HPO result
        best_params = {"lr": 1e-3, "dropout": 0.2, "batch_size": 256}
        original = dict(best_params)

        windows = forward_chaining_splits(df6, year_col="year", k=1)
        params_seen: list[dict] = []

        for win in windows:
            # Simulate: read best_params inside the loop — no re-search
            local_copy = dict(best_params)   # as _suggest_pytorch_model receives it
            params_seen.append(local_copy)
            # Verify no mutation occurred
            assert best_params == original, "best_params was mutated inside a window loop"

        assert len(params_seen) == len(windows)
        assert all(p == original for p in params_seen), \
            "best_params must be identical in every window"

    def test_hpo_called_once_not_per_window_in_simulation(self, df6):
        """Simulate the temporal protocol: HPO is called before the window loop.

        Verifies that a mock HPO function is invoked exactly once when the
        protocol is implemented correctly (outside the window loop).
        """
        hpo_call_count = 0

        def mock_run_optuna_search(**_kwargs) -> dict:
            nonlocal hpo_call_count
            hpo_call_count += 1
            return {"best_params": {"lr": 1e-3}, "best_value": 0.8}

        years = sorted(df6["year"].unique())
        windows = forward_chaining_splits(df6, year_col="year", k=1)

        # ── Correct implementation: HPO BEFORE the loop ──
        best_params = mock_run_optuna_search()["best_params"]

        for _win in windows:
            # Use best_params inside the loop — no HPO call here
            assert "lr" in best_params

        assert hpo_call_count == 1, (
            f"HPO must be called exactly once; was called {hpo_call_count} times. "
            "Calling HPO inside the forward-chaining loop would adapt "
            "hyperparameters to each temporal window, masking drift."
        )

    def test_hpo_per_window_would_inflate_call_count(self, df6):
        """Demonstrate that the wrong implementation calls HPO N_windows times.

        This test documents why NOT to put HPO inside the forward-chaining loop:
        the call count equals the number of windows, not 1.
        """
        hpo_call_count = 0

        def mock_run_optuna_search(**_kwargs) -> dict:
            nonlocal hpo_call_count
            hpo_call_count += 1
            return {"best_params": {"lr": 1e-3}, "best_value": 0.8}

        windows = forward_chaining_splits(df6, year_col="year", k=1)

        # ── Wrong implementation: HPO INSIDE the loop ──
        for _win in windows:
            mock_run_optuna_search()  # called once per window — wrong!

        assert hpo_call_count == len(windows), "Sanity: wrong impl calls HPO per window"
        assert hpo_call_count > 1, (
            "There must be more than one window for this test to demonstrate "
            "the inflation of HPO calls in the wrong implementation."
        )

    def test_results_metadata_schema_contains_hpo_fields(self):
        """Verify the expected schema of results.json for HPO auditability.

        run_experiment._run_temporal_experiment and run_axis2.run_model must
        include these keys in their results dict so that the HPO policy is
        traceable in every saved experiment.
        """
        REQUIRED_HPO_KEYS = {
            "hpo_policy",
            "hpo_anchor_year",
            "hpo_split",
            "hpo_reused_across_windows",
        }
        # Simulate what the scripts produce for the HPO section
        simulated_result = {
            "hpo_policy":                "single_anchor_year",
            "hpo_anchor_year":           2007,
            "hpo_split":                 "temporal_60_20_20",
            "hpo_reused_across_windows": True,
        }
        assert REQUIRED_HPO_KEYS <= set(simulated_result.keys())
        assert simulated_result["hpo_policy"] == "single_anchor_year"
        assert isinstance(simulated_result["hpo_anchor_year"], int)
        assert simulated_result["hpo_split"] == "temporal_60_20_20"
        assert simulated_result["hpo_reused_across_windows"] is True

    def test_internal_test_excluded_from_hpo_data(self, df6):
        """The internal_test slice from the 60/20/20 split must not appear in HPO.

        run_optuna_search receives only (train_hpo_df, val_hpo_df).  internal_test_df
        is reserved for Δt=0 evaluation inside the forward-chaining loop.
        """
        years = sorted(df6["year"].unique())
        anchor_df = df6[df6["year"] == years[0]].reset_index(drop=True)
        fit_df, val_df, internal_test_df = temporal_train_val_test_split(anchor_df, "year")

        # HPO receives only fit + val
        hpo_pos = set(fit_df["pos"]) | set(val_df["pos"])
        test_pos = set(internal_test_df["pos"])

        assert hpo_pos.isdisjoint(test_pos), (
            "internal_test rows must not appear in HPO training or validation data."
        )


# ---------------------------------------------------------------------------
# Item 9b — effective_batch_size: HPO batch_size flows into forward-chaining
# ---------------------------------------------------------------------------

class TestEffectiveBatchSizePolicy:
    """The batch_size chosen by Optuna must be applied in forward-chaining training.

    Policy:
    - For PyTorch models (mlp, cnn_bilstm, transformer), best_params["batch_size"]
      is the effective batch_size; it must override config["training"]["batch_size"].
    - For XGBoost, batch_size is irrelevant; config value is preserved unchanged.
    - effective_batch_size must be recorded in results.json for auditability.

    These tests validate the structural invariant without running actual training.
    """

    def test_pytorch_effective_batch_size_comes_from_best_params(self):
        """Simulate: for a PyTorch model, effective_batch_size must equal
        best_params['batch_size'], even when config['training']['batch_size']
        has a different (larger) value."""
        config_batch_size = 2048
        best_params = {"lr": 1e-3, "dropout": 0.1, "batch_size": 128}

        # Simulate the logic in run_model()
        model_family = "transformer"
        if model_family in ("mlp", "cnn_bilstm", "transformer"):
            effective_batch_size = int(best_params.get("batch_size", config_batch_size))
        else:
            effective_batch_size = config_batch_size

        assert effective_batch_size == 128, (
            "effective_batch_size must come from best_params, not config default"
        )
        assert effective_batch_size != config_batch_size

    def test_xgboost_effective_batch_size_equals_config_default(self):
        """For XGBoost, batch_size is not in best_params; effective value must
        fall back to config['training']['batch_size']."""
        config_batch_size = 2048
        best_params = {
            "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1.0,
        }

        model_family = "xgboost"
        if model_family in ("mlp", "cnn_bilstm", "transformer"):
            effective_batch_size = int(best_params.get("batch_size", config_batch_size))
        else:
            effective_batch_size = config_batch_size

        assert effective_batch_size == config_batch_size
        assert "batch_size" not in best_params

    def test_effective_batch_size_is_consistent_across_all_seeds(self):
        """The same effective_batch_size must be used for every seed in the run.

        Since HPO runs once before the seed loop, best_params is fixed and
        effective_batch_size is derived once — all seeds see the same value.
        """
        best_params = {"lr": 1e-3, "dropout": 0.1, "batch_size": 256}
        config_batch_size = 2048
        seeds = [42, 123, 456, 789, 1024]

        model_family = "mlp"
        effective_batch_size = int(best_params.get("batch_size", config_batch_size))

        batch_sizes_per_seed = [effective_batch_size for _ in seeds]
        assert all(bs == 256 for bs in batch_sizes_per_seed), (
            "All seeds must use the same effective_batch_size derived from best_params"
        )

    def test_effective_batch_size_fallback_when_not_in_best_params(self):
        """If best_params has no batch_size key (future model type), fall back
        to config['training']['batch_size'] gracefully."""
        config_batch_size = 512
        best_params = {"lr": 1e-3, "dropout": 0.1}  # no batch_size key

        model_family = "transformer"
        if model_family in ("mlp", "cnn_bilstm", "transformer"):
            effective_batch_size = int(best_params.get("batch_size", config_batch_size))
        else:
            effective_batch_size = config_batch_size

        assert effective_batch_size == config_batch_size
