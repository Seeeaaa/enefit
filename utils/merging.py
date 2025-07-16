import pandas as pd
from pandas import DataFrame
from utils.preprocessing import avg_weather_data
from pandas._typing import MergeHow


def merge_all_dfs(
    datasets: dict[str, DataFrame], how: MergeHow = "inner"
) -> DataFrame:
    """Merge all dataframes into one"""
    (
        train_df,
        gas_prices_df,
        client_df,
        electricity_prices_df,
        forecast_weather_df,
        historical_weather_df,
        station_county_mapping_df,
        # county_id_to_name_map,
        holidays_df,
    ) = (
        datasets["train"],
        datasets["gas_prices"],
        datasets["client"],
        datasets["electricity_prices"],
        datasets["forecast_weather"],
        datasets["historical_weather"],
        datasets["station_county_mapping"],
        # datasets["county_id_to_name_map"],
        datasets["holidays"],
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

    fp = "f1_"  # 1 Prefix for columns related to the 1 day forecast
    df = df.merge(
        avg_weather_data(
            forecast_weather_df, station_county_mapping_df
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
    hw_df = avg_weather_data(historical_weather_df, station_county_mapping_df)
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

    # Add different categories of holidays
    df = df.merge(
        holidays_df,
        how=how,
        on="date",
    ).fillna({c: 0 for c in holidays_df.columns.drop("date")})

    return df
