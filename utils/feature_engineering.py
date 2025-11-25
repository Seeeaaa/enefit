from pandas import DataFrame, Series, Timedelta, Timestamp, DateOffset
from pandas.core.groupby.generic import DataFrameGroupBy
import pandas as pd
import numpy as np


def get_lag(
    df: DataFrame,
    dt: str = "datetime",
    lag: int = 2,
    columns: list[str] = ["target"],
) -> DataFrame:
    """
    Shift 'dt' column by 'lag' days and rename the 'c' column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    dt : str
        Name of the datetime column to be shifted.
    lag : int
        Number of days to shift (must be 2 or greater).
    columns : list[str]
        List of columns to rename.

    Returns
    -------
    DataFrame
        DataFrame with the shifted datetime column and renamed target
        column.

    Raises
    ------
    ValueError
        If 'lag' is less than 2.
    KeyError
        If any column from 'columns' is not in the DataFrame.
    """
    if lag < 2:
        raise ValueError(f"'lag' must be at least 2 days, got {lag}")

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    return df.assign(**{dt: df[dt] + Timedelta(days=lag)}).rename(
        columns={col: f"{lag}d_lag_{col}" for col in columns}
    )


def get_moving_average(
    sorted_dfgb: DataFrameGroupBy,
    columns: list[str],
    window: int = 24,
    min_periods: int | None = None,
) -> DataFrame:
    """
    Compute rolling mean for specified columns of a grouped DataFrame
    and shift the datetime column by 48 hours.

    Parameters
    ----------
    sorted_dfgb : DataFrameGroupBy
        Grouped DataFrame (result of df.groupby(..., as_index=False)),
        where the original DataFrame was sorted by the datetime64[ns]
        index.
    columns : list[str]
        List of columns to aggregate.
    window : int
        Rolling window size in hours (min_periods=window).
    min_periods : int | None
        Minimum number of observations in the window required to have a
        value otherwise None.

    Returns
    -------
    DataFrame
        DataFrame containing:
        - all grouping columns,
        - the datetime column, shifted by 48 hours,
        - a new columns with the rolling mean.
    """
    txt = "h_ma_2d_lag_"
    missing = [c for c in columns if c not in sorted_dfgb.obj.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    # Store original dtypes
    original_dtypes = {
        f"{window}{txt}{c}": sorted_dfgb.obj[c].dtype for c in columns
    }
    df_rolled = (
        sorted_dfgb[columns]
        .rolling(
            Timedelta(f"{window} h"),
            min_periods=min_periods,
            closed="left",
        )
        .mean()
        .reset_index()
    )
    df_rolled.iloc[:, 0] += Timedelta(hours=48)  # Shift datetime
    df_rolled = df_rolled.rename(
        columns={c: f"{window}{txt}{c}" for c in columns}
    )
    df_rolled = df_rolled.astype(original_dtypes)
    return df_rolled


def add_dst_flag(df: DataFrame, datetime_col: str = "datetime") -> DataFrame:
    """
    Add a boolean 'dst' column indicating timestamps that fall within
    the predefined DST intervals for 2021-2023.
    """
    df["dst"] = (
        (df[datetime_col] < "2021-10-31 03:00:00")
        | (
            (df[datetime_col] >= "2022-03-27 03:00:00")
            & (df[datetime_col] < "2022-10-30 03:00:00")
        )
        | (df[datetime_col] >= "2023-03-26 03:00:00")
    )
    return df


def add_cyclic_datetime_features(
    df: DataFrame, datetime_col: str = "datetime", drop_raw: bool = True
) -> DataFrame:
    """
    Extract and encode cyclical datetime features.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame that contain datetime column.
    datetime_col : str
        Name of the datetime column to process.
    drop_raw : bool
        If True, drop the intermediate integer datetime columns (e.g.
        hour, weekday) after encoding.

    Returns
    -------
    DataFrame
        Same DataFrame with sin-cos features.

    Raises
    ------
    KeyError
        If 'datetime_col' is not in DataFrame.
    """
    if datetime_col not in df:
        raise KeyError(f"Column {datetime_col} not in DataFrame")
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])

    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter

    # Cyclic format
    for col, period in [
        ("hour", 24),
        ("weekday", 7),
        ("day_of_month", 30.4),
        ("month", 12),
        ("day_of_year", 365),
        ("week_of_year", 52),
        ("quarter", 4),
    ]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period).astype(
            "float32"
        )
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period).astype(
            "float32"
        )

    if drop_raw:
        df = df.drop(
            columns=[
                "hour",
                "weekday",
                "day_of_month",
                "month",
                "day_of_year",
                "week_of_year",
                "quarter",
            ]
        )

    return df


def get_split_bounds(
    dt: Series,
    train_share: float = 0.64,
    val_share: float = 0.16,
    n_val_splits: int = 3,
    n_test_splits: int = 3,
    fh: int = 7,
    expanding: bool = False,
) -> tuple[
    list[dict[str, tuple[pd.Timestamp, pd.Timestamp]]],
    list[dict[str, tuple[pd.Timestamp, pd.Timestamp]]],
]:
    """
    Compute time-based train-validation-test split boundaries for time
    series cross-validation.

    Parameters
    ----------
    dt : Series
        Series of datetime values (will be floored to days and
        deduplicated).
    train_share : float, default=0.64
        Proportion of the dataset to include in the training set.
    val_share : float, default=0.16
        Proportion of the dataset to include in the validation set.
    n_val_splits : int, default=3
        Number of validation splits to generate.
    n_test_splits : int, default=3
        Number of test splits to generate.
    fh : int, default=7
        Forecast horizon in days for each validation/test set.
    expanding : bool, default=False
        Whether the training window should expand in each split (True)
        or slide (False).

    Returns
    -------
    tuple of list of dict
        A tuple containing two lists:
        - Validation splits, each as a dict with 'train' and 'test'
        keys mapping to (start, end) tuples.
        - Test splits in the same format.

    Raises
    ------
    ValueError
        If train_share + val_share >= 1.0.
    """

    if train_share + val_share >= 1.0:
        raise ValueError(
            "train_share + val_share must be strictly less than 1.0"
        )
    dt = dt.dt.floor("d").drop_duplicates(ignore_index=True)  # type: ignore[attr-defined]

    train_start = dt.min()
    test_end = dt.max() + Timedelta(hours=23)

    total_days_interval = Timedelta(
        days=len(pd.date_range(train_start, test_end))
    )

    train_range = (total_days_interval * train_share).ceil("d")
    train_end = train_start + train_range - Timedelta(hours=1)  # exclude 00:00

    val_start = train_start + train_range
    val_range = (total_days_interval * val_share).ceil("d")
    val_end = val_start + val_range - Timedelta(hours=1)  # exclude 00:00

    test_start = val_start + val_range
    test_range = (
        test_end - test_start + Timedelta(hours=1)
    )  # include last 23 hours

    val_step = (val_range / n_val_splits).ceil("d")
    sub_val = []
    for i in range(n_val_splits):
        i_train_start = train_start + (not expanding) * i * val_step
        i_train_end = train_end + i * val_step
        i_val_start = val_start + i * val_step
        i_val_end = i_val_start + Timedelta(days=fh - 1, hours=23)

        sub_val.append(
            {
                "train": (i_train_start, i_train_end),
                "test": (
                    i_val_start,
                    i_val_end,
                ),
            }
        )

    test_step = (test_range / n_test_splits).ceil("d")

    sub_test = []
    for i in range(n_test_splits):
        i_train_start = train_start + int(not expanding) * (
            val_range + i * test_step
        )
        i_train_end = val_end + i * test_step
        i_test_start = test_start + i * test_step
        i_test_end = i_test_start + Timedelta(days=fh - 1, hours=23)

        sub_test.append(
            {
                "train": (i_train_start, i_train_end),
                "test": (i_test_start, i_test_end),
            }
        )

    return (
        sub_val,
        sub_test,
    )


def get_month_splits(
    start, train_range, test_range, shift, splits
) -> list[dict[str, tuple[Timestamp, Timestamp]]]:
    return [
        {
            "train": (
                start,
                start
                + DateOffset(months=train_range + shift * i)
                - Timedelta(hours=1),
            ),
            "test": (
                start + DateOffset(months=train_range + shift * i),
                start
                + DateOffset(months=train_range + test_range + shift * i)
                - Timedelta(hours=1),
            ),
        }
        for i in range(splits)
    ]
