"""Tests for evaluation: classification metrics, temporal metrics, statistical tests."""

from __future__ import annotations

import numpy as np
import pytest

from lift_nids.evaluation.metrics import _extract_f1_plus, compute_classification_metrics, f1_plus  # noqa: PLC2701
from lift_nids.evaluation.statistical_tests import (
    bootstrap_ci,
    friedman_nemenyi,
    wilcoxon_test,
)
from lift_nids.evaluation.temporal_metrics import (
    compute_naut,
    f1_plus_trajectory,
    median_trajectory,
)

# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        m = compute_classification_metrics(y, y)
        assert m["f1_macro"] == pytest.approx(1.0)
        assert m["f1_1"]     == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["f1_macro"] < 0.5
        assert m["f1_1"] == pytest.approx(0.0)

    def test_auc_metrics_with_proba(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_proba = rng.random(100)
        y_pred  = (y_proba >= 0.5).astype(int)
        m = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
        assert not np.isnan(m["auc_roc"])
        assert not np.isnan(m["auc_pr"])
        assert 0.0 <= m["auc_roc"] <= 1.0
        assert 0.0 <= m["auc_pr"]  <= 1.0

    def test_auc_nan_without_proba(self):
        y = np.array([0, 1])
        m = compute_classification_metrics(y, y)
        assert np.isnan(m["auc_roc"])
        assert np.isnan(m["auc_pr"])

    def test_all_keys_present(self):
        y = np.array([0, 1, 0, 1])
        m = compute_classification_metrics(y, y)
        expected_keys = {"f1_macro", "f1_0", "f1_1", "precision", "recall", "auc_roc", "auc_pr"}
        assert set(m.keys()) == expected_keys

    def test_single_class_in_y_true_does_not_crash(self):
        y_true = np.zeros(10, dtype=int)
        y_pred = np.zeros(10, dtype=int)
        m = compute_classification_metrics(y_true, y_pred)
        assert isinstance(m["f1_macro"], float)


class TestExtractF1Plus:
    """Regression guard: _extract_f1_plus must read 'f1_1', never fall back to 'f1_macro'."""

    def test_returns_f1_1_not_f1_macro(self):
        """Old code used f1_class_1 → f1_macro fallback; this must never happen."""
        metrics = {"f1_macro": 0.5, "f1_1": 0.0}
        assert _extract_f1_plus(metrics) == pytest.approx(0.0)

    def test_returns_f1_1_when_nonzero(self):
        metrics = {"f1_macro": 0.3, "f1_1": 0.9}
        assert _extract_f1_plus(metrics) == pytest.approx(0.9)

    def test_defaults_to_zero_when_f1_1_missing(self):
        metrics = {"f1_macro": 0.7}
        assert _extract_f1_plus(metrics) == pytest.approx(0.0)

    def test_consistent_with_compute_classification_metrics(self):
        y = np.array([0, 0, 1, 1])
        metrics = compute_classification_metrics(y, y)
        assert _extract_f1_plus(metrics) == pytest.approx(metrics["f1_1"])
        assert _extract_f1_plus(metrics) == pytest.approx(1.0)


class TestF1Plus:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1])
        assert f1_plus(y, y) == pytest.approx(1.0)

    def test_zero_tp(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert f1_plus(y_true, y_pred) == pytest.approx(0.0)

    def test_partial(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        # precision=1.0, recall=0.5 → F1=2/3
        assert f1_plus(y_true, y_pred) == pytest.approx(2 / 3, rel=1e-5)


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------

class TestF1PlusTrajectory:
    def _sample(self):
        return [
            (2010, 2011, 0.8), (2010, 2012, 0.7), (2010, 2013, 0.6),
            (2011, 2012, 0.75), (2011, 2013, 0.65),
        ]

    def test_groups_by_delta(self):
        traj = f1_plus_trajectory(self._sample())
        assert set(traj.keys()) == {1, 2, 3}

    def test_correct_values_for_delta1(self):
        traj = f1_plus_trajectory(self._sample())
        assert sorted(traj[1]) == pytest.approx(sorted([0.8, 0.75]))

    def test_empty_input(self):
        assert f1_plus_trajectory([]) == {}


class TestMedianTrajectory:
    def test_median(self):
        results = [(2010, 2011, 0.6), (2010, 2011, 0.8)]
        traj = median_trajectory(results)
        assert traj[1] == pytest.approx(0.7)

    def test_single_value(self):
        results = [(2010, 2011, 0.9)]
        traj = median_trajectory(results)
        assert traj[1] == pytest.approx(0.9)


class TestComputeNaut:
    """Tests for the IM28 trapezoidal nAUT formula.

    nAUT_H = (1/H) * Σ_{Δt=0}^{H-1} [ F̃1+(Δt) + F̃1+(Δt+1) ] / 2

    Δt=0 must be present for meaningful results; missing lags default to 0.0.
    """

    def test_perfect_trajectory_with_delta0(self):
        # All lags including Δt=0 are 1.0 → every trapezoid = 1.0
        traj = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        assert compute_naut(traj, H=3) == pytest.approx(1.0)

    def test_zero_trajectory(self):
        traj = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        assert compute_naut(traj, H=3) == pytest.approx(0.0)

    def test_partial_with_delta0(self):
        # traj = {0:1.0, 1:0.8, 2:0.6, 3:0.4}
        # nAUT_3 = [(1.0+0.8)/2 + (0.8+0.6)/2 + (0.6+0.4)/2] / 3
        traj = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        expected = ((1.0 + 0.8) / 2 + (0.8 + 0.6) / 2 + (0.6 + 0.4) / 2) / 3
        assert compute_naut(traj, H=3) == pytest.approx(expected)

    def test_missing_lags_treated_as_zero(self):
        # traj = {1: 0.9}; Δt=0 missing → 0.0, Δt=2,3 missing → 0.0
        # nAUT_3 = [(0+0.9)/2 + (0.9+0)/2 + (0+0)/2] / 3 = (0.45+0.45+0)/3 = 0.3
        traj = {1: 0.9}
        assert compute_naut(traj, H=3) == pytest.approx(0.3)

    def test_h1_trapezoidal(self):
        # nAUT_1 = (F(0)+F(1))/2
        traj = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
        assert compute_naut(traj, H=1) == pytest.approx((0.9 + 0.7) / 2)

    def test_h5_all_ones_with_delta0(self):
        traj = {t: 1.0 for t in range(6)}  # Δt=0..5 all 1.0
        assert compute_naut(traj, H=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_tuple(self):
        scores = np.array([0.7, 0.8, 0.75, 0.72, 0.78])
        ci = bootstrap_ci(scores)
        assert isinstance(ci, tuple) and len(ci) == 2

    def test_lower_le_upper(self):
        scores = np.array([0.7, 0.8, 0.75, 0.72, 0.78])
        lo, hi = bootstrap_ci(scores)
        assert lo <= hi

    def test_ci_contains_mean(self):
        scores = np.linspace(0.6, 0.9, 50)
        lo, hi = bootstrap_ci(scores, n_bootstrap=2000, seed=0)
        assert lo <= scores.mean() <= hi

    def test_wider_ci_for_less_confidence(self):
        scores = np.linspace(0.5, 1.0, 100)
        lo90, hi90 = bootstrap_ci(scores, confidence=0.90)
        lo99, hi99 = bootstrap_ci(scores, confidence=0.99)
        assert (hi99 - lo99) >= (hi90 - lo90)

    def test_constant_scores(self):
        scores = np.ones(10) * 0.8
        lo, hi = bootstrap_ci(scores)
        assert lo == pytest.approx(0.8)
        assert hi == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# wilcoxon_test
# ---------------------------------------------------------------------------

class TestWilcoxonTest:
    def test_identical_not_significant(self):
        scores = np.array([0.7, 0.8, 0.75, 0.72, 0.78, 0.80])
        # Identical arrays → no difference
        result = wilcoxon_test(scores, scores + 1e-9)
        assert isinstance(result["significant"], bool)
        assert isinstance(result["p_value"], float)

    def test_clearly_different_is_significant(self):
        a = np.array([0.9, 0.85, 0.88, 0.91, 0.87, 0.89, 0.92, 0.88, 0.90, 0.86])
        b = np.array([0.3, 0.25, 0.28, 0.31, 0.27, 0.29, 0.32, 0.28, 0.30, 0.26])
        result = wilcoxon_test(a, b)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_bonferroni_inflates_p(self):
        a = np.array([0.9, 0.85, 0.88, 0.91, 0.87, 0.89, 0.92, 0.88, 0.90, 0.86])
        b = a - 0.1
        r1 = wilcoxon_test(a, b, n_comparisons=1)
        r6 = wilcoxon_test(a, b, correction="bonferroni", n_comparisons=6)
        assert r6["p_adjusted"] >= r1["p_adjusted"]

    def test_output_keys(self):
        a = np.array([0.7, 0.8, 0.75, 0.72, 0.78])
        b = a + 0.05
        result = wilcoxon_test(a, b)
        assert {"statistic", "p_value", "p_adjusted", "significant"} == set(result.keys())

    def test_p_adjusted_capped_at_1(self):
        a = np.array([0.7, 0.8, 0.75, 0.72, 0.78])
        b = a + 0.05
        result = wilcoxon_test(a, b, correction="bonferroni", n_comparisons=1000)
        assert result["p_adjusted"] <= 1.0


# ---------------------------------------------------------------------------
# friedman_nemenyi
# ---------------------------------------------------------------------------

class TestFriedmanNemenyi:
    def _scores(self):
        rng = np.random.default_rng(42)
        # 10 datasets, 4 models; model 0 is clearly best
        base = rng.uniform(0.5, 0.9, (10, 4))
        base[:, 0] += 0.2
        return np.clip(base, 0, 1)

    def test_output_keys(self):
        result = friedman_nemenyi(self._scores(), ["A", "B", "C", "D"])
        assert {"friedman_stat", "p_value", "significant",
                "avg_ranks", "critical_difference", "nemenyi_significant"} == set(result.keys())

    def test_avg_ranks_cover_all_models(self):
        result = friedman_nemenyi(self._scores(), ["A", "B", "C", "D"])
        assert set(result["avg_ranks"].keys()) == {"A", "B", "C", "D"}

    def test_best_model_gets_lowest_rank(self):
        scores = self._scores()
        result = friedman_nemenyi(scores, ["best", "B", "C", "D"])
        assert result["avg_ranks"]["best"] == min(result["avg_ranks"].values())

    def test_avg_ranks_in_valid_range(self):
        # Ranks must lie in [1, k]; a consistently best model averages exactly 1.0
        scores = np.tile(np.array([0.9, 0.7, 0.5, 0.3]), (5, 1))
        result = friedman_nemenyi(scores, ["A", "B", "C", "D"])
        assert result["avg_ranks"] == {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

    def test_cd_positive(self):
        result = friedman_nemenyi(self._scores(), ["A", "B", "C", "D"])
        assert result["critical_difference"] > 0.0

    def test_nemenyi_matrix_shape(self):
        result = friedman_nemenyi(self._scores(), ["A", "B", "C", "D"])
        mat = result["nemenyi_significant"]
        assert mat.shape == (4, 4)

    def test_nemenyi_diagonal_false(self):
        result = friedman_nemenyi(self._scores(), ["A", "B", "C", "D"])
        mat = result["nemenyi_significant"]
        for name in ["A", "B", "C", "D"]:
            assert mat.loc[name, name] is False or mat.loc[name, name] == False  # noqa: E712

    def test_wrong_column_count_raises(self):
        with pytest.raises(ValueError, match="columns"):
            friedman_nemenyi(np.ones((5, 3)), ["A", "B"])


# ---------------------------------------------------------------------------
# cd_diagram (smoke test — no GUI assertion, just no crash)
# ---------------------------------------------------------------------------

class TestMaximalCliques:
    def test_overlapping_cliques_kept_separately(self):
        from lift_nids.evaluation.cd_diagram import _maximal_cliques

        # ranks 1..4 with CD ~2.1: {1,2,3} and {2,3,4} are distinct maximal
        # cliques; the {3,4} sub-clique is subsumed and must not be returned.
        assert _maximal_cliques([1.0, 2.0, 3.0, 4.0], 2.098) == [(0, 2), (1, 3)]

    def test_no_cliques_when_cd_small(self):
        from lift_nids.evaluation.cd_diagram import _maximal_cliques

        assert _maximal_cliques([1.0, 2.0, 3.0, 4.0], 0.5) == []

    def test_single_global_clique_when_cd_large(self):
        from lift_nids.evaluation.cd_diagram import _maximal_cliques

        assert _maximal_cliques([1.0, 2.0, 3.0, 4.0], 3.5) == [(0, 3)]


class TestCDDiagram:
    def test_smoke_no_output_path(self):
        import matplotlib
        matplotlib.use("Agg")
        from lift_nids.evaluation.cd_diagram import plot_cd_diagram

        avg_ranks = {"Transformer": 1.5, "CNN-BiLSTM": 2.3, "MLP": 2.9, "XGBoost": 3.3}
        plot_cd_diagram(avg_ranks, critical_difference=0.85)

    def test_saves_to_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from lift_nids.evaluation.cd_diagram import plot_cd_diagram

        out = tmp_path / "cd.png"
        avg_ranks = {"A": 1.2, "B": 2.5, "C": 3.3}
        plot_cd_diagram(avg_ranks, critical_difference=0.9, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
