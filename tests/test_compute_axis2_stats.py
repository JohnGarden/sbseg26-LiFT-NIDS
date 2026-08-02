"""Tests for scripts/compute_axis2_stats.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compute_axis2_stats import ALL_MODELS, SEEDS, compute_stats, load_seed_scores

# Per-model base values chosen so transformer > cnn_bilstm > xgboost > mlp
# for every seed (all pairwise differences share one sign at n=5, which
# yields the minimum one-sided Wilcoxon p = 1/2^5 = 0.03125).
_BASE = {"xgboost": 0.38, "mlp": 0.28, "cnn_bilstm": 0.44, "transformer": 0.57}


def _make_tree(root: Path) -> None:
    """Create a synthetic axis2 results tree with 4 models x 5 seeds."""
    for model, base in _BASE.items():
        d = root / model / "k_cumulative"
        d.mkdir(parents=True)
        for i, seed in enumerate(SEEDS):
            payload = {"seed": seed, "nAUT_1": base + 0.001 * i}
            (d / f"seed_{seed}.json").write_text(json.dumps(payload))


def test_load_seed_scores_missing_file_raises(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    (tmp_path / "mlp" / "k_cumulative" / "seed_42.json").unlink()
    with pytest.raises(FileNotFoundError, match="seed_42"):
        load_seed_scores(tmp_path, "mlp", "k_cumulative", SEEDS, "nAUT_1")


def test_load_seed_scores_missing_metric_raises(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    with pytest.raises(KeyError, match="nAUT_9"):
        load_seed_scores(tmp_path, "mlp", "k_cumulative", SEEDS, "nAUT_9")


def test_compute_stats_structure_and_values(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    stats = compute_stats(tmp_path)

    assert stats["models"] == ALL_MODELS
    assert stats["seeds"] == SEEDS
    assert stats["metric"] == "nAUT_1"

    # With a fully consistent ranking over 5 blocks and 4 models the Friedman
    # statistic is exactly n * (k - 1) = 15 and p ~= 0.0018.
    fr = stats["friedman"]
    assert fr["statistic"] == pytest.approx(15.0)
    assert fr["p_value"] == pytest.approx(0.0018, abs=2e-4)
    assert fr["significant"] is True
    assert fr["avg_ranks"]["transformer"] == 1.0
    assert fr["avg_ranks"]["mlp"] == 4.0

    pairs = stats["wilcoxon_one_sided"]["pairs"]
    assert len(pairs) == 6
    primary = [p for p in pairs if p["role"] == "primary_prespecified"]
    assert len(primary) == 1
    assert primary[0]["greater"] == "transformer"
    assert primary[0]["lesser"] == "cnn_bilstm"
    # Minimum attainable one-sided p at n=5.
    assert primary[0]["p_value"] == pytest.approx(0.03125)
    assert primary[0]["p_bonferroni_6"] == pytest.approx(0.1875)


def test_main_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tree(tmp_path)
    import compute_axis2_stats as mod

    monkeypatch.setattr(
        "sys.argv", ["compute_axis2_stats.py", "--output-dir", str(tmp_path)]
    )
    assert mod.main() == 0

    out = json.loads((tmp_path / "stats_axis2.json").read_text())
    assert out["friedman"]["statistic"] == pytest.approx(15.0)
    assert "timestamp" in out
