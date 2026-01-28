import pandas as pd
from pandas import DataFrame
from pandas._typing import MergeHow
from utils.preprocessing import avg_weather_data


def merge_all_dfs(
    datasets: dict[str, DataFrame], how: MergeHow = "left"
) -> DataFrame:
    """Merge all dataframes into one"""
    (
        train_df,
        gas_prices_df,
        client_df,
        electricity_prices_df,
        forecast_weather_df,
        historical_weather_df,
        weather_station_to_county_mapping_df,
        # holidays_df,
    ) = (
        datasets["train"],
        datasets["gas_prices"],
        datasets["client"],
        datasets["electricity_prices"],
        datasets["forecast_weather"],
        datasets["historical_weather"],
        datasets["weather_station_to_county_mapping"],
    )

    df = pd.merge(
        left=train_df,
        right=client_df.drop(columns=["date"]),
        how=how,
        on=["county", "product_type", "is_business", "data_block_id"],
    )

    df = df.merge(
        right=gas_prices_df,
        how=how,
        on=["data_block_id"],
    )

    df = df.merge(
        right=electricity_prices_df,
        how=how,
        left_on=["datetime", "data_block_id"],
        right_on=["electricity_datetime", "data_block_id"],
    ).drop(columns=["electricity_datetime"])

    fp = "f1_"  # Prefix for columns related to the 1 day forecast
    df = df.merge(
        avg_weather_data(
            forecast_weather_df, weather_station_to_county_mapping_df
        ).add_prefix(fp),
        how=how,
        left_on=["county", "datetime", "data_block_id"],
        right_on=[
            fp + c for c in ["county", "forecast_datetime", "data_block_id"]
        ],
    ).drop(
        columns=[
            fp + c
            for c in [
                "county",
                "origin_datetime",
                "hours_ahead",
                "forecast_datetime",
                "data_block_id",
            ]
        ]
    )

    hp = "h2_"  # Prefix for columns related to 2 day historical data
    hw_df = avg_weather_data(
        historical_weather_df, weather_station_to_county_mapping_df
    )
    hw_df["fully_available_at"] = hw_df["datetime"] + pd.Timedelta("2 d")
    df = df.merge(
        hw_df.add_prefix(hp),
        how=how,
        left_on=["county", "datetime"],
        right_on=[hp + c for c in ["county", "fully_available_at"]],
    ).drop(
        columns=[
            hp + c
            for c in [
                "county",
                "datetime",
                "fully_available_at",
                "data_block_id",
            ]
        ]
    )

    if "holidays" in datasets:
        holidays_df = datasets["holidays"]
        holidays_names = holidays_df.columns.drop(["date"])
        df = df.merge(
            holidays_df,
            how=how,
            on="date",
        ).fillna({column: False for column in holidays_names})
        df[holidays_names] = df[holidays_names].astype("bool")

    df = df.drop(columns=["data_block_id", "date"])

    return df
