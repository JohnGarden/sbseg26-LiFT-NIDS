"""Tests for src/lift_nids/data/preprocessing.py."""

import unittest

import numpy as np
import pandas as pd

from lift_nids.data.preprocessing import FlowPreprocessor, infer_feature_types


def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feat_a": rng.exponential(scale=10, size=n),  # right-skewed
            "feat_b": rng.uniform(0, 100, size=n),
            "feat_c": rng.integers(0, 5, size=n).astype(float),
            "category": rng.choice(["tcp", "udp", "icmp", "arp", "other"], size=n),
            "label": rng.integers(0, 2, size=n),
        }
    )


class TestFlowPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _make_df()
        self.num_cols = ["feat_a", "feat_b", "feat_c"]
        self.cat_cols = ["category"]

    def test_fit_transform_shape(self) -> None:
        pp = FlowPreprocessor()
        result = pp.fit_transform(self.df, self.num_cols, self.cat_cols)
        # 3 numerical + 4 one-hot columns (top-4 categories from 5, "other" might drop)
        self.assertEqual(result.shape[0], len(self.df))
        self.assertGreater(result.shape[1], len(self.num_cols))

    def test_numerical_range_after_transform(self) -> None:
        pp = FlowPreprocessor()
        result = pp.fit_transform(self.df, self.num_cols)
        num_result = result[self.num_cols].to_numpy()
        self.assertGreaterEqual(num_result.min(), -1e-6)
        self.assertLessEqual(num_result.max(), 1.0 + 1e-6)

    def test_transform_on_unseen_data(self) -> None:
        pp = FlowPreprocessor()
        pp.fit(self.df, self.num_cols, self.cat_cols)
        new_df = _make_df(n=20, seed=99)
        result = pp.transform(new_df)
        self.assertEqual(result.shape[0], 20)

    def test_no_label_leakage(self) -> None:
        """Label column should not appear in transformed output."""
        pp = FlowPreprocessor()
        result = pp.fit_transform(self.df, self.num_cols, self.cat_cols)
        self.assertNotIn("label", result.columns)

    def test_no_nan_after_transform(self) -> None:
        """Transformed output must contain no NaN values."""
        pp = FlowPreprocessor()
        result = pp.fit_transform(self.df, self.num_cols, self.cat_cols)
        self.assertFalse(result.isnull().any().any())

    def test_fit_uses_only_train_data(self) -> None:
        """Scaler parameters must come solely from the training split."""
        train_df = self.df.iloc[:80]
        test_df = self.df.iloc[80:]

        pp = FlowPreprocessor()
        pp.fit(train_df, self.num_cols)

        # Capture scaler state after fit
        data_min_after_fit = pp._scaler.data_min_.copy()
        data_max_after_fit = pp._scaler.data_max_.copy()

        # transform() must not modify scaler parameters
        pp.transform(test_df)

        np.testing.assert_array_equal(pp._scaler.data_min_, data_min_after_fit)
        np.testing.assert_array_equal(pp._scaler.data_max_, data_max_after_fit)

    def test_save_load_pickle_roundtrip(self) -> None:
        import tempfile
        from pathlib import Path

        pp = FlowPreprocessor()
        pp.fit(self.df, self.num_cols, self.cat_cols)
        orig_result = pp.transform(self.df)

        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = Path(tmp) / "preprocessor.pkl"
            pp.save_pickle(pkl_path)
            pp2 = FlowPreprocessor.load_pickle(pkl_path)
        loaded_result = pp2.transform(self.df)
        np.testing.assert_allclose(
            orig_result[self.num_cols].to_numpy(),
            loaded_result[self.num_cols].to_numpy(),
            atol=1e-5,
        )

    def test_save_load_roundtrip(self, tmp_path: str = None) -> None:
        import tempfile

        pp = FlowPreprocessor()
        pp.fit(self.df, self.num_cols, self.cat_cols)
        orig_result = pp.transform(self.df)

        with tempfile.TemporaryDirectory() as tmp:
            pp.save(tmp)
            pp2 = FlowPreprocessor.load(tmp)
        loaded_result = pp2.transform(self.df)
        np.testing.assert_allclose(
            orig_result[self.num_cols].to_numpy(),
            loaded_result[self.num_cols].to_numpy(),
            atol=1e-5,
        )


class TestInferFeatureTypes(unittest.TestCase):
    def test_separates_categorical(self) -> None:
        df = _make_df()
        num_cols, cat_cols = infer_feature_types(df, label_col="label")
        self.assertIn("category", cat_cols)
        self.assertNotIn("label", num_cols)
        self.assertNotIn("label", cat_cols)

    def test_all_numerical_no_categories(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": [0, 1]})
        num_cols, cat_cols = infer_feature_types(df, label_col="label")
        self.assertEqual(cat_cols, [])
        self.assertIn("a", num_cols)


if __name__ == "__main__":
    unittest.main()
