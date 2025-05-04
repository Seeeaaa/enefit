import pandas as pd
from pandas import DataFrame
from notebooks.utils.preprocessing import avg_weather_data

def merge_all_dfs(
    data: tuple[dict[str, DataFrame], dict[str, DataFrame]],
) -> DataFrame:
    # def merge_all_dfs(
    #     original: dict[str, DataFrame], additional: dict[str, DataFrame]
    # ) -> DataFrame:
    """Merge all dataframes into one"""
    (
        train_df,
        gas_prices_df,
        client_df,
        electricity_prices_df,
        forecast_weather_df,
        historical_weather_df,
        station_county_mapping_df,
        county_id_to_name_map,
    ) = (
        data[0]["train"],
        data[0]["gas_prices"],
        data[0]["client"],
        data[0]["electricity_prices"],
        data[0]["forecast_weather"],
        data[0]["historical_weather"],
        data[0]["station_county_mapping"],
        data[0]["county_id_to_name_map"],
    )
    holidays_df = data[1]["holidays"]

    # # Drop spring NaNs and impute autumn NaNs with interpolated values
    # na_datetimes = train_df[train_df.isna().any(axis=1)]["datetime"].unique()
    # df = train_df.loc[~train_df["datetime"].isin(na_datetimes[1::2])].assign(
    #     target=lambda x: x["target"].interpolate()
    # )

    df = pd.merge(
        left=train_df,
        right=client_df.drop(columns=["date"]),
        how="inner",  # save dtype
        on=["county", "product_type", "is_business", "data_block_id"],
    )

    df = df.merge(
        right=gas_prices_df,
        how="inner",
        on=["data_block_id"],
    )

    df = df.merge(
        right=electricity_prices_df,
        how="inner",
        left_on=["datetime", "data_block_id"],
        right_on=["electricity_datetime", "data_block_id"],
    ).drop(columns=["electricity_datetime"])

    fp = "f1_"  # 1 Prefix for columns related to the 1 day forecast
    df = df.merge(
        avg_weather_data(
            forecast_weather_df, station_county_mapping_df
        ).add_prefix(fp),
        # how="left",
        how="inner",
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
        # how="left",
        how="inner",
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

    # Add a flag indicating Daylight Saving Time
    df["dst"] = ~(
        (
            (df.datetime >= "2021-10-31 03:00:00")
            & (df.datetime < "2022-03-27 03:00:00")
        )
        | (
            (df.datetime >= "2022-10-30 03:00:00")
            & (df.datetime < "2023-03-26 03:00:00")
        )
    )

    # Add different categories of holidays
    df = (
        df.merge(holidays_df, how="left", on=["date"])
        .fillna({"holiday_type": "ordinary_day"})
        .astype({"holiday_type": "category"})
    )

    return df
