from pandas import DataFrame, Series, Timedelta, Timestamp, DateOffset
from pandas.core.groupby.generic import SeriesGroupBy
import pandas as pd
import numpy as np


def get_lag(
    df: pd.DataFrame,
    value_col: str,
    lag: str,
    datetime_col: str,
) -> pd.DataFrame:
    """
    Shift the datetime column by a given time offset and rename the
    value column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_col : str
        Name of the column to be lagged and renamed.
    lag : str
        Time offset compatible with pandas.Timedelta (e.g. '48h', '7d').
    datetime_col : str
        Name of the datetime column to be shifted.

    Returns
    -------
    pd.DataFrame
        DataFrame with the shifted datetime column and the renamed
        lagged column.
    """
    return df.assign(
        **{datetime_col: df[datetime_col] + pd.Timedelta(lag)}
    ).rename(columns={value_col: f"{value_col}_lag_{lag}"})


def add_lags(
    df: pd.DataFrame,
    value_col: str,
    lags: list[str],
    datetime_col: str,
    group_cols: list[str],
) -> pd.DataFrame:

    id_cols = [datetime_col] + group_cols
    original_df = df[id_cols + [value_col]]

    for lag in lags:
        df = df.merge(
            get_lag(
                original_df,
                value_col,
                lag,
                datetime_col,
            ),
            how="left",
            on=id_cols,
            validate="1:1",
        )
    return df


def prepare_time_series_groupby(
    df: pd.DataFrame,
    value_col: str,
    datetime_col: str,
    group_cols: list[str],
) -> SeriesGroupBy:
    """
    Prepare a SeriesGroupBy with a DatetimeIndex, suitable for
    time-based rolling operations.
    """
    return (
        df.sort_values(group_cols + [datetime_col])
        .set_index(datetime_col)
        .groupby(
            by=group_cols,
            sort=False,
            observed=True,
        )[value_col]
    )


def compute_rolling_features(
    sgb: SeriesGroupBy,
    value_col: str,
    window: str,
    funcs: list[str],
) -> pd.DataFrame:
    """
    Compute rolling aggregations over a time-based window and return a
    flat DataFrame with named features.
    """
    return (
        sgb.rolling(
            window,
            closed="left",
        )
        .agg(funcs)
        .astype("float32")
        .rename(
            columns={
                func: f"{value_col}_{window}_win_{func}" for func in funcs
            }
        )
        .reset_index()
    )


# def add_lag(
#     df: DataFrame,
#     datetime_column: str = "datetime",
#     lag_in_days: int = 2,
#     id_columns: list[str] = [
#         "county",
#         "product_type",
#         "is_business",
#         "is_consumption",
#         "datetime",
#     ],
#     target_columns: list[str] = ["target"],
# ) -> DataFrame:
#     df = df.copy(deep=True)
#     shifted_df = df[id_columns + target_columns].copy(deep=True)
#     shifted_df[datetime_column] = shifted_df[datetime_column] + Timedelta(
#         days=lag_in_days
#     )
#     shifted_df = shifted_df.rename(
#         columns={c: f"{lag_in_days}d_lag_{c}" for c in target_columns}
#     )
#     df = df.merge(shifted_df, how="left", on=id_columns, validate="1:1")
#     return df


# def get_moving_average(
#     sorted_dfgb: DataFrameGroupBy,
#     columns: list[str],
#     window: int = 24,
#     min_periods: int | None = None,
# ) -> DataFrame:
#     """
#     Compute rolling mean for specified columns of a grouped DataFrame
#     and shift the datetime column by 48 hours.

#     Parameters
#     ----------
#     sorted_dfgb : DataFrameGroupBy
#         Grouped DataFrame (result of df.groupby(..., as_index=False)),
#         where the original DataFrame was sorted by the datetime64[ns]
#         index.
#     columns : list[str]
#         List of columns to aggregate.
#     window : int
#         Rolling window size in hours (min_periods=window).
#     min_periods : int | None
#         Minimum number of observations in the window required to have a
#         value otherwise None.

#     Returns
#     -------
#     DataFrame
#         DataFrame containing:
#         - all grouping columns,
#         - the datetime column, shifted by 48 hours,
#         - a new columns with the rolling mean.
#     """
#     txt = "h_ma_2d_lag_"
#     missing = [c for c in columns if c not in sorted_dfgb.obj.columns]
#     if missing:
#         raise KeyError(f"Columns not found in DataFrame: {missing}")

#     # Store original dtypes
#     original_dtypes = {
#         f"{window}{txt}{c}": sorted_dfgb.obj[c].dtype for c in columns
#     }
#     df_rolled = (
#         sorted_dfgb[columns]
#         .rolling(
#             Timedelta(f"{window} h"),
#             min_periods=min_periods,
#             closed="left",
#         )
#         .mean()
#         .reset_index()
#     )
#     df_rolled.iloc[:, 0] += Timedelta(hours=48)  # Shift datetime
#     df_rolled = df_rolled.rename(
#         columns={c: f"{window}{txt}{c}" for c in columns}
#     )
#     df_rolled = df_rolled.astype(original_dtypes)
#     return df_rolled


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
    start: Timestamp,
    train_range: int,
    test_range: int,
    shift: int,
    splits: int,
) -> list[dict[str, tuple[Timestamp, Timestamp]]]:
    """
    Create sliding month-based train/test splits.

    For each split, the training window begins at 'start' and spans
    'train_range' months, followed immediately by a test window of
    'test_range' months. For subsequent splits ('splits' > 1), the
    train/test boundary is shifted forward by 'shift' months for each
    split.

    End timestamps are inclusive and truncated by one hour, i.e. each
    interval starts at 00:00 and ends at 23:00.

    Parameters
    ----------
    start : pandas.Timestamp
        Start timestamp of the first training window.
    train_range : int
        Number of months in the initial training window.
    test_range : int
        Number of months in each test window.
    shift : int
        Number of months by which the train/test boundary is shifted
        forward for each subsequent split.
    splits : int
        Number of train/test splits to generate.

    Returns
    -------
    splits : list[dict[str, tuple[pandas.Timestamp, pandas.Timestamp]]]
        List of splits. Each split is a dictionary with keys "train"
        and "test", where values are (start, end) timestamp tuples
        defining inclusive time intervals.
    """

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


def drop_split(
    df: DataFrame, bounds: tuple, to_drop: list
) -> tuple[DataFrame, Series]:
    """
    Filters a DataFrame by datetime bounds, drops specified columns,
    and splits into features and target.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing 'datetime' and 'target' columns.
    bounds : tuple
        A tuple specifying the datetime range to filter df.
    to_drop : list
        List of column names to drop from the df.

    Returns
    -------
    X : DataFrame
        DataFrame with specified columns dropped and 'target' removed.
    y : Series
        Target values corresponding to the 'target' column in the
        filtered DataFrame.
    """
    start, end = bounds[0], bounds[1]
    subset = df[(df["datetime"] >= start) & (df["datetime"] <= end)].drop(
        to_drop, axis=1
    )
    X, y = subset.drop(["target"], axis=1), subset["target"]
    return X, y
