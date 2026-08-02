"""Flow feature preprocessing: log-transform, min-max scaling, frequency-limited one-hot.

Pipeline validated by FlowTransformer (IM09):
  1. log1p transform on numerical features (handles right-skewed traffic metrics)
  2. min-max scaling to [0, 1]
  3. frequency-limited one-hot for categorical features (drops rare categories)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FlowPreprocessor:
    """Stateful preprocessor — call fit() on training data, transform() on all splits."""

    def __init__(self, max_categories: int = 50, log_transform: bool = True) -> None:
        self.max_categories = max_categories
        self.log_transform = log_transform

        self.numerical_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self._category_maps: dict[str, list[str]] = {}
        self._scaler = MinMaxScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        categorical_cols: list[str] | None = None,
    ) -> FlowPreprocessor:
        """Fit on training DataFrame. Returns self."""
        self.numerical_cols = list(numerical_cols)
        self.categorical_cols = list(categorical_cols or [])

        # Frequency-limited categories: keep top-k most frequent
        for col in self.categorical_cols:
            top_cats = (
                df[col].value_counts().head(self.max_categories).index.tolist()
            )
            self._category_maps[col] = top_cats

        # Log-transform then fit scaler on numerical columns.
        # Use float32 numpy array + in-place ops to avoid the 2x float64 allocation
        # that pandas .copy().clip() triggers on large datasets.
        num_data = df[self.numerical_cols].to_numpy(dtype=np.float32, copy=True)
        np.clip(num_data, 0, None, out=num_data)
        num_data[~np.isfinite(num_data)] = 0.0
        if self.log_transform:
            np.log1p(num_data, out=num_data)
        self._scaler.fit(num_data)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing. Returns new DataFrame."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        result = pd.DataFrame(index=df.index)

        # Numerical: log1p + min-max (float32 + in-place to avoid 2x float64 allocation)
        num_data = df[self.numerical_cols].to_numpy(dtype=np.float32, copy=True)
        np.clip(num_data, 0, None, out=num_data)
        num_data[~np.isfinite(num_data)] = 0.0
        if self.log_transform:
            np.log1p(num_data, out=num_data)
        scaled = self._scaler.transform(num_data)
        result[self.numerical_cols] = scaled

        # Categorical: frequency-limited one-hot
        for col in self.categorical_cols:
            known = self._category_maps[col]
            masked = df[col].where(df[col].isin(known), other="__other__")
            dummies = pd.get_dummies(masked, prefix=col)
            # Align to training columns (add missing, drop unseen)
            for cat in known:
                col_name = f"{col}_{cat}"
                if col_name not in dummies:
                    dummies[col_name] = 0
            train_cols = [f"{col}_{c}" for c in known]
            result = pd.concat([result, dummies[train_cols].astype(np.float32)], axis=1)

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        categorical_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(df, numerical_cols, categorical_cols).transform(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pickle(self, path: str | Path) -> None:
        """Serialize preprocessor state to a pickle file."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle(cls, path: str | Path) -> FlowPreprocessor:
        """Deserialize preprocessor from a pickle file."""
        import pickle

        with open(Path(path), "rb") as f:
            return pickle.load(f)

    def save(self, path: str | Path) -> None:
        """Save preprocessor state to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "max_categories": self.max_categories,
            "log_transform": self.log_transform,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "category_maps": self._category_maps,
            "scaler_scale": self._scaler.scale_.tolist(),
            "scaler_min": self._scaler.min_.tolist(),
            "scaler_data_min": self._scaler.data_min_.tolist(),
            "scaler_data_max": self._scaler.data_max_.tolist(),
        }
        (path / "preprocessor.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FlowPreprocessor:
        """Load preprocessor state from a directory."""
        path = Path(path)
        meta = json.loads((path / "preprocessor.json").read_text())
        obj = cls(
            max_categories=meta["max_categories"],
            log_transform=meta["log_transform"],
        )
        obj.numerical_cols = meta["numerical_cols"]
        obj.categorical_cols = meta["categorical_cols"]
        obj._category_maps = meta["category_maps"]
        obj._scaler = MinMaxScaler()
        n = len(meta["scaler_scale"])
        obj._scaler.scale_ = np.array(meta["scaler_scale"])
        obj._scaler.min_ = np.array(meta["scaler_min"])
        obj._scaler.data_min_ = np.array(meta["scaler_data_min"])
        obj._scaler.data_max_ = np.array(meta["scaler_data_max"])
        obj._scaler.data_range_ = obj._scaler.data_max_ - obj._scaler.data_min_
        obj._scaler.feature_names_in_ = np.array(obj.numerical_cols)
        obj._scaler.n_features_in_ = n
        obj._scaler.n_samples_seen_ = 0
        obj._fitted = True
        return obj


def infer_feature_types(
    df: pd.DataFrame,
    label_col: str | None = None,
    id_cols: list[str] | None = None,
    max_unique_for_categorical: int = 50,
) -> tuple[list[str], list[str]]:
    """Heuristically split columns into numerical and categorical lists.

    Returns (numerical_cols, categorical_cols).
    """
    if label_col is not None and label_col not in df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    exclude = set(filter(None, [label_col] + (id_cols or [])))
    numerical, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= max_unique_for_categorical:
                # Low-cardinality integers treated as numerical (flag columns)
                numerical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)
    return numerical, categorical
