"""Tests for src/lift_nids/data/splits.py."""

import unittest

import numpy as np
import pandas as pd

from lift_nids.data.splits import (
    ForwardChainingWindow,
    forward_chaining_splits,
    iter_forward_chaining_windows,
    static_split,
    temporal_train_val_test_split,
)


def _make_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feat": rng.random(n),
            "label": rng.integers(0, 2, n),
        }
    )


def _make_temporal_df(years: list[int], rows_per_year: int = 100) -> pd.DataFrame:
    rows = []
    for y in years:
        for _ in range(rows_per_year):
            rows.append({"year": y, "feat": float(y), "label": int(y % 2)})
    return pd.DataFrame(rows)


class TestStaticSplit(unittest.TestCase):
    def test_sizes_sum_to_total(self) -> None:
        df = _make_df(300)
        train, val, test = static_split(df, 0.6, 0.2, 0.2, seed=42)
        self.assertEqual(len(train) + len(val) + len(test), len(df))

    def test_approximate_ratios(self) -> None:
        df = _make_df(3000)
        train, val, test = static_split(df, 0.6, 0.2, 0.2, seed=42)
        self.assertAlmostEqual(len(train) / len(df), 0.6, delta=0.02)
        self.assertAlmostEqual(len(val) / len(df), 0.2, delta=0.02)
        self.assertAlmostEqual(len(test) / len(df), 0.2, delta=0.02)

    def test_no_overlap(self) -> None:
        df = _make_df(300)
        train, val, test = static_split(df, seed=42)
        idx_train = set(train.index)
        idx_val = set(val.index)
        idx_test = set(test.index)
        self.assertEqual(len(idx_train & idx_val), 0)
        self.assertEqual(len(idx_train & idx_test), 0)
        self.assertEqual(len(idx_val & idx_test), 0)

    def test_stratified_split_preserves_ratio(self) -> None:
        df = _make_df(600)
        train, val, test = static_split(df, 0.6, 0.2, 0.2, stratify_col="label", seed=42)
        orig_ratio = df["label"].mean()
        self.assertAlmostEqual(train["label"].mean(), orig_ratio, delta=0.05)
        self.assertAlmostEqual(test["label"].mean(), orig_ratio, delta=0.05)

    def test_reproducible_with_same_seed(self) -> None:
        df = _make_df(300)
        train1, _, _ = static_split(df, seed=42)
        train2, _, _ = static_split(df, seed=42)
        pd.testing.assert_frame_equal(train1, train2)

    def test_different_seeds_differ(self) -> None:
        df = _make_df(300)
        train1, _, _ = static_split(df, seed=42)
        train2, _, _ = static_split(df, seed=99)
        self.assertFalse(train1.index.equals(train2.index))


class TestForwardChainingSplits(unittest.TestCase):
    def setUp(self) -> None:
        self.years = list(range(2007, 2013))  # 6 years: 2007..2012
        self.df = _make_temporal_df(self.years, rows_per_year=50)

    # ── window count ──────────────────────────────────────────────────────────

    def test_k1_number_of_windows(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=1)
        # 6 years → 5 windows (each trains on 1 year, last year has no future)
        self.assertEqual(len(windows), 5)

    def test_k2_number_of_windows(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=2)
        # 6 years → 4 windows (each trains on 2 consecutive years)
        self.assertEqual(len(windows), 4)

    def test_k3_number_of_windows(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=3)
        # 6 years → 3 windows
        self.assertEqual(len(windows), 3)

    def test_cumulative_number_of_windows(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k="cumulative")
        # 6 years → 5 windows (grows from [2007] to [2007..2011])
        self.assertEqual(len(windows), 5)

    # ── return type ───────────────────────────────────────────────────────────

    def test_returns_forward_chaining_window_objects(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=1)
        for w in windows:
            self.assertIsInstance(w, ForwardChainingWindow)

    # ── k=1 first-window composition ─────────────────────────────────────────

    def test_k1_first_window_train_years(self) -> None:
        w = forward_chaining_splits(self.df, "year", k=1)[0]
        self.assertEqual(w.train_years, [2007])
        self.assertEqual(w.anchor_year, 2007)

    def test_k1_first_window_test_years(self) -> None:
        w = forward_chaining_splits(self.df, "year", k=1)[0]
        self.assertEqual(sorted(w.test_dfs_by_year.keys()), [2008, 2009, 2010, 2011, 2012])

    # ── k=2 first-window composition ─────────────────────────────────────────

    def test_k2_first_window_train_and_test_years(self) -> None:
        w = forward_chaining_splits(self.df, "year", k=2)[0]
        self.assertEqual(w.train_years, [2007, 2008])
        self.assertEqual(w.anchor_year, 2008)
        self.assertEqual(sorted(w.test_dfs_by_year.keys()), [2009, 2010, 2011, 2012])

    # ── k=3 first-window composition ─────────────────────────────────────────

    def test_k3_first_window_train_and_test_years(self) -> None:
        w = forward_chaining_splits(self.df, "year", k=3)[0]
        self.assertEqual(w.train_years, [2007, 2008, 2009])
        self.assertEqual(w.anchor_year, 2009)
        self.assertEqual(sorted(w.test_dfs_by_year.keys()), [2010, 2011, 2012])

    # ── no leakage ───────────────────────────────────────────────────────────

    def test_k1_no_overlap_train_test_years(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=1)
        for w in windows:
            train_set = set(w.train_years)
            test_set = set(w.test_dfs_by_year.keys())
            self.assertEqual(len(train_set & test_set), 0)

    def test_k2_no_overlap_train_test_years(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=2)
        for w in windows:
            train_set = set(w.train_years)
            test_set = set(w.test_dfs_by_year.keys())
            self.assertEqual(len(train_set & test_set), 0)

    def test_cumulative_never_includes_future_in_train(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k="cumulative")
        for w in windows:
            max_train = max(w.train_years)
            for test_yr in w.test_dfs_by_year:
                self.assertGreater(test_yr, max_train)

    # ── cumulative window grows ───────────────────────────────────────────────

    def test_cumulative_train_grows(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k="cumulative")
        train_sizes = [len(w.train_df) for w in windows]
        for i in range(1, len(train_sizes)):
            self.assertGreater(train_sizes[i], train_sizes[i - 1])

    # ── per-year data integrity ───────────────────────────────────────────────

    def test_test_df_contains_only_its_year(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=1)
        for w in windows:
            for yr, yr_df in w.test_dfs_by_year.items():
                self.assertTrue((yr_df["year"] == yr).all())

    def test_train_df_contains_only_train_years(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k=2)
        for w in windows:
            actual = set(w.train_df["year"].unique())
            self.assertEqual(actual, set(w.train_years))

    # ── integration: unique (anchor_year, test_year) pairs ───────────────────

    def test_k1_f1plus_pairs_are_unique(self) -> None:
        """Simulates f1plus_records structure: each (anchor, test_year) is unique."""
        windows = forward_chaining_splits(self.df, "year", k=1)
        pairs: list[tuple[int, int]] = []
        for w in windows:
            for test_yr in w.test_dfs_by_year:
                pairs.append((w.anchor_year, test_yr))
        self.assertEqual(len(pairs), len(set(pairs)), "Duplicate (anchor, test_year) pairs found")

    def test_cumulative_f1plus_pairs_are_unique(self) -> None:
        windows = forward_chaining_splits(self.df, "year", k="cumulative")
        pairs: list[tuple[int, int]] = []
        for w in windows:
            for test_yr in w.test_dfs_by_year:
                pairs.append((w.anchor_year, test_yr))
        self.assertEqual(len(pairs), len(set(pairs)), "Duplicate (anchor, test_year) pairs found")


class TestIterForwardChainingWindows(unittest.TestCase):
    def setUp(self) -> None:
        self.years = list(range(2007, 2013))  # 6 years: 2007..2012
        self.df = _make_temporal_df(self.years, rows_per_year=50)

    # ── is a lazy iterator, not a list ───────────────────────────────────────

    def test_returns_iterator_with_next(self) -> None:
        result = iter_forward_chaining_windows(self.df, "year", k=1)
        self.assertTrue(hasattr(result, "__iter__"))
        self.assertTrue(hasattr(result, "__next__"))

    def test_is_not_a_list(self) -> None:
        result = iter_forward_chaining_windows(self.df, "year", k=1)
        self.assertNotIsInstance(result, list)

    # ── window count matches eager version ───────────────────────────────────

    def test_k1_produces_5_windows(self) -> None:
        windows = list(iter_forward_chaining_windows(self.df, "year", k=1))
        self.assertEqual(len(windows), 5)

    def test_k2_produces_4_windows(self) -> None:
        windows = list(iter_forward_chaining_windows(self.df, "year", k=2))
        self.assertEqual(len(windows), 4)

    def test_k3_produces_3_windows(self) -> None:
        windows = list(iter_forward_chaining_windows(self.df, "year", k=3))
        self.assertEqual(len(windows), 3)

    def test_cumulative_produces_5_windows(self) -> None:
        windows = list(iter_forward_chaining_windows(self.df, "year", k="cumulative"))
        self.assertEqual(len(windows), 5)

    # ── ForwardChainingWindow objects ─────────────────────────────────────────

    def test_produces_forward_chaining_window_objects(self) -> None:
        for win in iter_forward_chaining_windows(self.df, "year", k=1):
            self.assertIsInstance(win, ForwardChainingWindow)

    # ── train_years and anchor_year match eager for all k ────────────────────

    def test_train_years_match_eager_for_all_k(self) -> None:
        for k in [1, 2, 3, "cumulative"]:
            lazy = list(iter_forward_chaining_windows(self.df, "year", k=k))
            eager = forward_chaining_splits(self.df, "year", k=k)
            for lw, ew in zip(lazy, eager):
                self.assertEqual(lw.train_years, ew.train_years, f"k={k}")
                self.assertEqual(lw.anchor_year, ew.anchor_year, f"k={k}")

    # ── test_years match eager for all k ─────────────────────────────────────

    def test_test_years_match_eager_for_all_k(self) -> None:
        for k in [1, 2, 3, "cumulative"]:
            lazy = list(iter_forward_chaining_windows(self.df, "year", k=k))
            eager = forward_chaining_splits(self.df, "year", k=k)
            for lw, ew in zip(lazy, eager):
                self.assertEqual(
                    sorted(lw.test_dfs_by_year.keys()),
                    sorted(ew.test_dfs_by_year.keys()),
                    f"k={k}",
                )

    # ── data integrity ────────────────────────────────────────────────────────

    def test_train_df_contains_only_train_years(self) -> None:
        for win in iter_forward_chaining_windows(self.df, "year", k=2):
            actual = set(win.train_df["year"].unique())
            self.assertEqual(actual, set(win.train_years))

    def test_test_df_contains_only_its_year(self) -> None:
        for win in iter_forward_chaining_windows(self.df, "year", k=1):
            for yr, yr_df in win.test_dfs_by_year.items():
                self.assertTrue((yr_df["year"] == yr).all())

    def test_no_train_test_year_overlap(self) -> None:
        for win in iter_forward_chaining_windows(self.df, "year", k=1):
            train_set = set(win.train_years)
            test_set = set(win.test_dfs_by_year.keys())
            self.assertEqual(len(train_set & test_set), 0)

    # ── generator semantics ───────────────────────────────────────────────────

    def test_supports_early_stop_via_close(self) -> None:
        gen = iter_forward_chaining_windows(self.df, "year", k=1)
        first = next(gen)
        self.assertIsNotNone(first.train_df)
        gen.close()  # Must not raise

    def test_cumulative_train_grows_monotonically(self) -> None:
        sizes = [
            len(w.train_df)
            for w in iter_forward_chaining_windows(self.df, "year", k="cumulative")
        ]
        for i in range(1, len(sizes)):
            self.assertGreater(sizes[i], sizes[i - 1])

    # ── F1+ pair uniqueness (methodological invariant) ───────────────────────

    def test_f1plus_pairs_unique_k1(self) -> None:
        pairs = [
            (win.anchor_year, yr)
            for win in iter_forward_chaining_windows(self.df, "year", k=1)
            for yr in win.test_dfs_by_year
        ]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_f1plus_pairs_unique_cumulative(self) -> None:
        pairs = [
            (win.anchor_year, yr)
            for win in iter_forward_chaining_windows(self.df, "year", k="cumulative")
            for yr in win.test_dfs_by_year
        ]
        self.assertEqual(len(pairs), len(set(pairs)))


class TestTemporalTrainValTestSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.years = list(range(2007, 2013))
        self.df = _make_temporal_df(self.years, rows_per_year=100)

    def test_sizes_sum_to_total(self) -> None:
        fit, val, test = temporal_train_val_test_split(self.df, "year")
        self.assertEqual(len(fit) + len(val) + len(test), len(self.df))

    def test_approximate_fractions(self) -> None:
        fit, val, test = temporal_train_val_test_split(
            self.df, "year", train_frac=0.60, val_frac=0.20
        )
        n = len(self.df)
        self.assertAlmostEqual(len(fit) / n, 0.60, delta=0.02)
        self.assertAlmostEqual(len(val) / n, 0.20, delta=0.02)
        self.assertAlmostEqual(len(test) / n, 0.20, delta=0.02)

    def test_temporal_order_fit_before_val_before_test(self) -> None:
        fit, val, test = temporal_train_val_test_split(self.df, "year")
        # All fit years ≤ all val years ≤ all test years
        self.assertLessEqual(fit["year"].max(), val["year"].min())
        self.assertLessEqual(val["year"].max(), test["year"].min())

    def test_no_row_overlap(self) -> None:
        # Add a unique row_id so we can check identity after reset_index.
        df_with_id = self.df.copy()
        df_with_id["_rid"] = range(len(df_with_id))
        fit, val, test = temporal_train_val_test_split(df_with_id, "year")
        ids_fit  = set(fit["_rid"])
        ids_val  = set(val["_rid"])
        ids_test = set(test["_rid"])
        self.assertEqual(len(ids_fit & ids_val), 0)
        self.assertEqual(len(ids_fit & ids_test), 0)
        self.assertEqual(len(ids_val & ids_test), 0)
        self.assertEqual(ids_fit | ids_val | ids_test, set(range(len(df_with_id))))

    def test_cumulative_train_df_grows_and_order_preserved(self) -> None:
        """After patch: iter_forward_chaining_windows uses isin() not pd.concat."""
        windows = list(iter_forward_chaining_windows(self.df, "year", k="cumulative"))
        sizes = []
        for win in windows:
            fit, val, itest = temporal_train_val_test_split(win.train_df, "year")
            sizes.append(len(fit) + len(val) + len(itest))
            # temporal order preserved inside each window
            combined_years = (
                list(fit["year"]) + list(val["year"]) + list(itest["year"])
            )
            self.assertEqual(combined_years, sorted(combined_years))
        for i in range(1, len(sizes)):
            self.assertGreater(sizes[i], sizes[i - 1])

    def test_train_years_and_anchor_unchanged_after_patch(self) -> None:
        """Replacing pd.concat with isin must not alter train_years or anchor_year."""
        for k in [1, 2, 3, "cumulative"]:
            lazy = list(iter_forward_chaining_windows(self.df, "year", k=k))
            eager = forward_chaining_splits(self.df, "year", k=k)
            for lw, ew in zip(lazy, eager):
                self.assertEqual(lw.train_years, ew.train_years, f"k={k}")
                self.assertEqual(lw.anchor_year, ew.anchor_year, f"k={k}")
                self.assertEqual(
                    sorted(lw.test_dfs_by_year.keys()),
                    sorted(ew.test_dfs_by_year.keys()),
                    f"k={k}",
                )


if __name__ == "__main__":
    unittest.main()
