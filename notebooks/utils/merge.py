import pandas as pd
from utils.process import avg_weather_data


def merge_all_dfs(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    ) = (
        dfs["train"],
        dfs["gas_prices"],
        dfs["client"],
        dfs["electricity_prices"],
        dfs["forecast_weather"],
        dfs["historical_weather"],
        dfs["station_county_mapping"],
        # dfs["county_id_to_name_map"],
    )

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
    # df["dst"] = ~(
    #     ((df.datetime >= na_datetimes[0]) & (df.datetime < na_datetimes[1]))
    #     | ((df.datetime >= na_datetimes[2]) & (df.datetime < na_datetimes[3]))
    # )

    return df

    #     # estonia_holidays = holidays.EE(years=range(2021, 2024), language='en_US')
    #     # for date, name in estonia_holidays.items():
    #     #     print(date, name)


# https://www.timeanddate.com/holidays/estonia/2021
# Holidays and Observances in Estonia in 2021
# Date	 	Name	Type
# 1 Jan	Friday	New Year's Day	National Holiday
# 6 Jan	Wednesday	Epiphany	Observance
# 2 Feb	Tuesday	Anniversary of Tartu Peace Treaty	Observance
# 24 Feb	Wednesday	Independence Day	National Holiday
# 14 Mar	Sunday	Mother Tongue Day	Observance
# 20 Mar	Saturday	March Equinox	Season
# 2 Apr	Friday	Good Friday	National Holiday
# 4 Apr	Sunday	Easter Sunday	National Holiday
# 1 May	Saturday	Labor Day	National Holiday
# 9 May	Sunday	Mothers' Day	Observance
# 23 May	Sunday	Pentecost	National Holiday
# 4 Jun	Friday	Flag Day	Observance
# 14 Jun	Monday	Day of Mourning	Observance
# 21 Jun	Monday	June Solstice	Season
# 23 Jun	Wednesday	Victory Day	National Holiday
# 24 Jun	Thursday	Midsummer Day	National Holiday
# 20 Aug	Friday	Independence Restoration Day	National Holiday
# 23 Aug	Monday	Day of Remembrance for Victims of Communism and Nazism	Observance
# 12 Sep	Sunday	Grandparents' Day	Observance
# 22 Sep	Wednesday	Resistance Day	Observance
# 22 Sep	Wednesday	September Equinox	Season
# 16 Oct	Saturday	Finno-Ugric Day	Observance
# 2 Nov	Tuesday	All Soul's Day	Observance
# 14 Nov	Sunday	Father's Day	Observance
# 16 Nov	Tuesday	Day of Declaration of Sovereignty	Observance
# 21 Dec	Tuesday	December Solstice	Season
# 24 Dec	Friday	Christmas Eve	National Holiday
# 25 Dec	Saturday	Christmas Day	National Holiday
# 26 Dec	Sunday	Boxing Day	National Holiday
