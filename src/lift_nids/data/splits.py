"""Train/validation/test split strategies.

Two protocols:
  - static: random stratified split (CICIoT2023 sanity check, 60/20/20)
  - forward_chaining: temporal, train on k consecutive years, evaluate each
    future year individually (MAWIFlow primary axis)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

# Type alias for k parameter
KType = int | Literal["cumulative"]


@dataclass
class ForwardChainingWindow:
    """One training window with per-year test slices for forward-chaining evaluation.

    Attributes:
        train_df: All rows belonging to the training years.
        test_dfs_by_year: Maps each test year to the rows for that year.
        train_years: Years included in the training window.
        anchor_year: Last year of the training window (y_train for F1+ computation).
    """

    train_df: pd.DataFrame
    test_dfs_by_year: dict[int, pd.DataFrame]
    train_years: list[int]
    anchor_year: int




def static_split(
    df: pd.DataFrame,
    train_size: float = 0.60,
    val_size: float = 0.20,
    test_size: float = 0.20,
    stratify_col: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random stratified split into train / val / test.

    Returns (train_df, val_df, test_df).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "Sizes must sum to 1."

    stratify = df[stratify_col] if stratify_col else None
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=stratify,
        random_state=seed,
    )
    # Split temp into val and test; adjust stratify to temp index
    strat_temp = temp_df[stratify_col] if stratify_col else None
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        stratify=strat_temp,
        random_state=seed,
    )
    return train_df, val_df, test_df


def forward_chaining_splits(
    df: pd.DataFrame,
    year_col: str,
    k: KType = 1,
) -> list[ForwardChainingWindow]:
    """Generate forward-chaining windows for temporal evaluation.

    ``k`` is the training window size (number of consecutive years to train on).
    For each valid window, ALL subsequent years are evaluated individually,
    enabling correct computation of F1+(y_train, y_test) at each lag
    Δt = test_year − anchor_year.

    Args:
        df: DataFrame with a column indicating the year of each flow.
        year_col: Name of the column containing the year integer.
        k: Training window size in years, or ``"cumulative"``.
            - int k: sliding window of exactly k consecutive years; all later
              years become individual test sets.
            - ``"cumulative"``: training window grows from year[0]; all years
              after the current window are individual test sets.

    Returns:
        List of :class:`ForwardChainingWindow` objects, one per valid window.

    Examples:
        k=1, years 2007–2012::

            Window 0: train=[2007], test_dfs_by_year={2008:…, 2009:…, …, 2012:…}
            Window 1: train=[2008], test_dfs_by_year={2009:…, …, 2012:…}
            …

        k=2, years 2007–2012::

            Window 0: train=[2007, 2008], test_dfs_by_year={2009:…, …, 2012:…}
            Window 1: train=[2008, 2009], test_dfs_by_year={2010:…, …, 2012:…}
            …

        cumulative, years 2007–2012::

            Window 0: train=[2007],        test_dfs_by_year={2008:…, …, 2012:…}
            Window 1: train=[2007, 2008],  test_dfs_by_year={2009:…, …, 2012:…}
            …
    """
    years = sorted(int(y) for y in df[year_col].unique())
    windows: list[ForwardChainingWindow] = []

    if k == "cumulative":
        for i in range(len(years) - 1):
            train_yrs = years[: i + 1]
            test_yrs = years[i + 1 :]
            windows.append(
                ForwardChainingWindow(
                    train_df=df[df[year_col].isin(train_yrs)],
                    test_dfs_by_year={y: df[df[year_col] == y] for y in test_yrs},
                    train_years=train_yrs,
                    anchor_year=train_yrs[-1],
                )
            )
    else:
        k_int = int(k)
        for i in range(len(years) - k_int):
            train_yrs = years[i : i + k_int]
            test_yrs = years[i + k_int :]
            windows.append(
                ForwardChainingWindow(
                    train_df=df[df[year_col].isin(train_yrs)],
                    test_dfs_by_year={y: df[df[year_col] == y] for y in test_yrs},
                    train_years=train_yrs,
                    anchor_year=train_yrs[-1],
                )
            )

    return windows


def iter_forward_chaining_windows(
    df: pd.DataFrame,
    year_col: str,
    k: KType = 1,
    is_sorted: bool = False,
) -> Generator[ForwardChainingWindow, None, None]:
    """Lazy generator yielding one ForwardChainingWindow at a time.

    Semantics are identical to :func:`forward_chaining_splits` but
    memory-efficient: pre-indexes ``df_by_year`` once (O(n) DataFrames where n
    is the number of distinct years) and yields one window on demand so the
    caller can release each window before the next is materialised.

    Args:
        df: DataFrame with a column indicating the year of each flow.
        year_col: Name of the column containing the year integer.
        k: Training window size in years, or ``"cumulative"``.
        is_sorted: If True, ``df`` is already sorted by ``year_col`` (then by
            timestamp within each year).  Enables zero-copy ``iloc`` slicing
            instead of boolean-indexed copies for both train and test splits.

    Yields:
        :class:`ForwardChainingWindow` instances in chronological order.
    """
    years = sorted(int(y) for y in df[year_col].unique())

    if is_sorted:
        # Compute integer boundaries once; all per-year slices become views.
        vc = df[year_col].value_counts(sort=False)
        starts: dict[int, int] = {}
        ends: dict[int, int] = {}
        acc = 0
        for y in years:
            starts[y] = acc
            acc += int(vc[y])
            ends[y] = acc
        df_by_year: dict[int, pd.DataFrame] = {
            y: df.iloc[starts[y] : ends[y]] for y in years
        }
    else:
        df_by_year = {y: df[df[year_col] == y] for y in years}

    if k == "cumulative":
        for i in range(len(years) - 1):
            train_yrs = years[: i + 1]
            test_yrs = years[i + 1 :]
            if is_sorted:
                train_df = df.iloc[0 : ends[train_yrs[-1]]]
            else:
                train_df = pd.concat(
                    [df_by_year[y] for y in train_yrs], ignore_index=True
                )
            yield ForwardChainingWindow(
                train_df=train_df,
                test_dfs_by_year={y: df_by_year[y] for y in test_yrs},
                train_years=train_yrs,
                anchor_year=train_yrs[-1],
            )
    else:
        k_int = int(k)
        for i in range(len(years) - k_int):
            train_yrs = years[i : i + k_int]
            test_yrs = years[i + k_int :]
            if is_sorted:
                train_df = df.iloc[starts[train_yrs[0]] : ends[train_yrs[-1]]]
            else:
                train_df = pd.concat(
                    [df_by_year[y] for y in train_yrs], ignore_index=True
                )
            yield ForwardChainingWindow(
                train_df=train_df,
                test_dfs_by_year={y: df_by_year[y] for y in test_yrs},
                train_years=train_yrs,
                anchor_year=train_yrs[-1],
            )


def temporal_train_val_test_split(
    df: pd.DataFrame,
    year_col: str,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    timestamp_col: str | None = None,
    already_sorted: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a training-window DataFrame temporally into fit / val / internal_test.

    Rows are sorted by ``year_col`` then, when ``timestamp_col`` is provided,
    by the timestamp within each year.  The three splits are strictly ordered
    in time: fit rows < val rows < internal_test rows.

    ``internal_test_df`` is the held-out slice used for Δt=0 evaluation:
    performance on data from the same time window as training but never seen
    during fitting or early-stopping.  It is NOT training-set evaluation.

    Args:
        df: Training window DataFrame (may span multiple years for k>1).
        year_col: Column containing the integer year of each row.
        train_frac: Fraction of rows for model fitting (default 0.60).
        val_frac: Fraction of rows for validation / early stopping (default 0.20).
            The remainder (1 − train_frac − val_frac) becomes internal_test.
        timestamp_col: Optional column for within-year ordering.  If provided
            and absent from ``df``, raises ``ValueError``.
        already_sorted: If True, ``df`` is already sorted by ``(year_col,
            timestamp_col)`` and the sort step is skipped, saving one full copy.

    Returns:
        ``(fit_df, val_df, internal_test_df)`` — disjoint DataFrames in temporal
        order; fit rows strictly precede val rows, which strictly precede
        internal_test rows.
    """
    if already_sorted:
        sorted_df = df
    else:
        if timestamp_col is not None and timestamp_col not in df.columns:
            raise ValueError(
                f"timestamp_col={timestamp_col!r} not found in DataFrame columns: "
                f"{list(df.columns)}"
            )
        sort_keys = [year_col] + ([timestamp_col] if timestamp_col else [])
        sorted_df = df.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    n = len(sorted_df)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * (1.0 - train_frac - val_frac)))
    n_fit = max(0, n - n_val - n_test)
    return (
        sorted_df.iloc[:n_fit].reset_index(drop=True),
        sorted_df.iloc[n_fit : n_fit + n_val].reset_index(drop=True),
        sorted_df.iloc[n_fit + n_val :].reset_index(drop=True),
    )


def extract_year(df: pd.DataFrame, timestamp_col: str) -> pd.Series:
    """Extract year from a timestamp column (converts to datetime if needed)."""
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    return ts.dt.year
