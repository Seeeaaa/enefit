from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
import pandas as pd
import numpy as np


def get_lag(df: DataFrame, dt: str, lag: int, c: str) -> DataFrame:
    """
    Shift 'dt' column by 'lag' days and rename the 'c' column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    dt : str
        Name of the datetime column to be shifted.
    lag : int
        Number of days to shift.
    c : str
        Name of the column to rename.

    Returns
    -------
    DataFrame
        DataFrame with the shifted datetime column and renamed target
        column.

    Raises
    ------
    ValueError
        If 'lag' is less than 2.
    """
    if lag < 2:
        raise ValueError(f"'lag' must be at least 2 days, got {lag}")
    return df.assign(**{dt: df[dt] + pd.Timedelta(days=lag)}).rename(
        columns={c: f"{lag}d_lag_{c}"}
    )


def get_moving_average(
    dfgb: DataFrameGroupBy,
    c: str,
    window: int,
) -> DataFrame:
    """
    Compute rolling mean for a specified column of a grouped DataFrame
    and shift the datetime column by 48 hours.

    Parameters
    ----------
    dfgb : DataFrameGroupBy
        Grouped DataFrame (result of df.groupby(..., as_index=False)),
        where the original DataFrame was sorted by the datetime index.
    c : str
        Name of the column to aggregate.
    window : int
        Rolling window size in hours (min_periods=window).

    Returns
    -------
    DataFrame
        DataFrame containing:
        - all grouping columns,
        - the datetime column, shifted by 48 h,
        - a new column with the rolling mean.
    """

    return (
        dfgb[c]
        .rolling(pd.Timedelta(f"{window}h"), min_periods=window, closed="left")
        .mean()
        .reset_index()
        .pipe(
            lambda x: x.assign(
                **{x.columns[0]: x[x.columns[0]] + pd.Timedelta(hours=48)}
            )
        )
        .rename(columns={c: f"{window}h_ma_2d_lag_{c}"})
    )


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
        ("day_of_year", 365.25),
        ("week_of_year", 52),
        ("quarter", 4),
    ]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

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
