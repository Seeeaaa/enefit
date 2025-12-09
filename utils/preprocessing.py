import pandas as pd
from pandas import DataFrame, Series
import numpy as np


def process_train(df: DataFrame) -> DataFrame:
    df = df.copy()
    # Drop spring NaNs and impute autumn NaNs with interpolated values
    na_datetimes = df[df.isna().any(axis=1)]["datetime"].unique()
    df = df.loc[~df["datetime"].isin(na_datetimes[1::2])].assign(
        target=lambda x: x["target"].interpolate()
    )
    df = df[
        [
            "county",
            "product_type",
            "is_business",
            "is_consumption",
            "datetime",
            "target",
            "data_block_id",
        ]
    ].astype(
        {
            "county": "category",
            "product_type": "category",
            "is_business": "bool",
            "is_consumption": "bool",
            "datetime": "datetime64[ns]",
            "target": "float32",
            "data_block_id": "uint16",
        }
    )
    df["date"] = df["datetime"].dt.date  # type: ignore[attr-defined]
    return df


def process_gas_prices(df: DataFrame) -> DataFrame:
    return df[
        [
            "data_block_id",
            "lowest_price_per_mwh",
            "highest_price_per_mwh",
        ]
    ].astype(
        {
            "data_block_id": "uint16",
            "lowest_price_per_mwh": "float32",
            "highest_price_per_mwh": "float32",
        }
    )


def process_client(df: DataFrame) -> DataFrame:
    return df[
        [
            "county",
            "product_type",
            "is_business",
            "date",
            "eic_count",
            "installed_capacity",
            "data_block_id",
        ]
    ].astype(
        {
            "county": "category",
            "product_type": "category",
            "is_business": "bool",
            "date": "datetime64[ns]",
            "eic_count": "uint32",
            "installed_capacity": "float32",
            "data_block_id": "uint16",
        }
    )


def process_electricity_prices(df: DataFrame) -> DataFrame:
    return df.astype(
        {
            "origin_date": "datetime64[ns]",
            "euros_per_mwh": "float32",
            "data_block_id": "uint16",
        }
    ).assign(
        electricity_datetime=lambda x: x["origin_date"] + pd.Timedelta(2, "d")
    )[
        ["electricity_datetime", "euros_per_mwh", "data_block_id"]
    ]


def process_forecast_weather(df: DataFrame) -> DataFrame:
    df[["latitude", "longitude"]] = (
        df[["latitude", "longitude"]].round(1).mul(10)
    )

    df["hours_ahead"] = pd.to_timedelta(df["hours_ahead"], "h")

    precipitation_threshold = 0.1  # Threshold in mm
    df["snowfall_mm"] = df["snowfall"] * 1000  # To mm
    df["total_precipitation_mm"] = df["total_precipitation"] * 1000  # To mm
    df[["snowfall_mm", "total_precipitation_mm"]] = df[
        ["snowfall_mm", "total_precipitation_mm"]
    ].mask(
        df[["snowfall_mm", "total_precipitation_mm"]].abs()
        < precipitation_threshold,
        0,
    )

    df[
        [
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "cloudcover_total",
        ]
    ] = (
        df[
            [
                "cloudcover_low",
                "cloudcover_mid",
                "cloudcover_high",
                "cloudcover_total",
            ]
        ]
        .round(2)
        .mul(100)
    )

    df = df.rename(
        columns={
            "10_metre_u_wind_component": "u_component",
            "10_metre_v_wind_component": "v_component",
        }
    )
    df["windspeed"] = np.sqrt(df["u_component"] ** 2 + df["v_component"] ** 2)

    df[["direct_solar_radiation", "surface_solar_radiation_downwards"]] = df[
        ["direct_solar_radiation", "surface_solar_radiation_downwards"]
    ].clip(lower=0)

    df = df[
        [
            "latitude",
            "longitude",
            "origin_datetime",
            "hours_ahead",
            "forecast_datetime",
            "data_block_id",
            "temperature",
            "dewpoint",
            "snowfall_mm",
            "total_precipitation_mm",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "cloudcover_total",
            "u_component",
            "v_component",
            "windspeed",
            "direct_solar_radiation",
            "surface_solar_radiation_downwards",
        ]
    ].astype(
        {
            "latitude": "uint16",
            "longitude": "uint16",
            "origin_datetime": "datetime64[ns]",
            "forecast_datetime": "datetime64[ns]",
            "data_block_id": "uint16",
            "temperature": "float32",
            "dewpoint": "float32",
            "snowfall_mm": "float32",
            "total_precipitation_mm": "float32",
            "cloudcover_low": "uint8",
            "cloudcover_mid": "uint8",
            "cloudcover_high": "uint8",
            "cloudcover_total": "uint8",
            "u_component": "float32",
            "v_component": "float32",
            "windspeed": "float32",
            "direct_solar_radiation": "float32",
            "surface_solar_radiation_downwards": "float32",
        }
    )

    return df


def process_historical_weather(df: DataFrame) -> DataFrame:
    hw_to_drop = [1176339, 1176343]  # Drop duplicates
    df = df.drop(index=hw_to_drop).reset_index(drop=True)
    df[["latitude", "longitude"]] = (
        df[["latitude", "longitude"]].round(1).mul(10)
    )

    snow_water_density_ratio = 0.1
    df["snowfall"] = df["snowfall"] * 10 * snow_water_density_ratio
    df = df.rename(
        columns={
            "snowfall": "snowfall_mm",
            "rain": "rain_mm",
            "windspeed_10m": "windspeed",
        }
    )
    df["winddirection_10m"] = np.deg2rad(df["winddirection_10m"])
    df["u_component"] = -df["windspeed"] * np.sin(df["winddirection_10m"])
    df["v_component"] = -df["windspeed"] * np.cos(df["winddirection_10m"])

    df = df[
        [
            "latitude",
            "longitude",
            "datetime",
            "data_block_id",
            "temperature",
            "dewpoint",
            "snowfall_mm",
            "rain_mm",
            "surface_pressure",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "cloudcover_total",
            "windspeed",
            "u_component",
            "v_component",
            "shortwave_radiation",
            "direct_solar_radiation",
            "diffuse_radiation",
        ]
    ].astype(
        {
            "latitude": "uint16",
            "longitude": "uint16",
            "datetime": "datetime64[ns]",
            "data_block_id": "uint16",
            "temperature": "float32",
            "dewpoint": "float32",
            "snowfall_mm": "float32",
            "rain_mm": "float32",
            "surface_pressure": "float32",
            "cloudcover_low": "uint8",
            "cloudcover_mid": "uint8",
            "cloudcover_high": "uint8",
            "cloudcover_total": "uint8",
            "windspeed": "float32",
            "u_component": "float32",
            "v_component": "float32",
            "shortwave_radiation": "uint16",
            "direct_solar_radiation": "uint16",
            "diffuse_radiation": "uint16",
        }
    )
    return df


def process_station_county_mapping(df: DataFrame) -> DataFrame:
    df = df[
        [
            "latitude",
            "longitude",
            "county_name",
            "county",
        ]
    ]

    df[["latitude", "longitude"]] = (
        df[["latitude", "longitude"]].round(1).mul(10)
    )

    df[["county_name", "county"]] = df[["county_name", "county"]].fillna(
        {"county_name": "unknown", "county": 12}
    )

    df["county_name"] = df["county_name"].str.lower()
    return (
        df.astype(
            {
                "latitude": "uint16",
                "longitude": "uint16",
                "county_name": "category",
                "county": "uint8",
            }
        )
        .astype({"county": "category"})
        .sort_values(["latitude", "longitude"], ignore_index=True)
    )


def process_county_id_to_name_map(mapper: Series) -> Series:
    return mapper.str.lower()


def process_original_dfs(
    data: dict[str, DataFrame | Series],
) -> dict[str, DataFrame | Series]:
    return {
        "train": process_train(data["train"]),
        "gas_prices": process_gas_prices(data["gas_prices"]),
        "client": process_client(data["client"]),
        "electricity_prices": process_electricity_prices(
            data["electricity_prices"]
        ),
        "forecast_weather": process_forecast_weather(data["forecast_weather"]),
        "historical_weather": process_historical_weather(
            data["historical_weather"]
        ),
        "station_county_mapping": process_station_county_mapping(
            data["station_county_mapping"]
        ),
        "county_id_to_name_map": process_county_id_to_name_map(
            data["county_id_to_name_map"]
        ),
    }


def process_additional_dfs(
    data: dict[str, DataFrame],
) -> dict[str, DataFrame]:
    # holidays
    columns = data["holidays"].holiday_type.unique()
    data["holidays"] = (
        pd.crosstab(
            data["holidays"]["date"],
            data["holidays"]["holiday_type"],
        )
        .rename_axis(columns=None)
        .reset_index()
        .astype({c: "category" for c in columns})
    )
    return data


def process_all_dfs(
    datasets: dict[str, DataFrame | Series],
) -> dict[str, DataFrame | Series]:
    # datasets = datasets.copy()
    additional = datasets.pop("holidays", None)
    processed_original = process_original_dfs(datasets)

    if isinstance(additional, DataFrame):
        holidays_dict = {"holidays": additional}
        processed_additional = process_additional_dfs(holidays_dict)
        return processed_original | processed_additional
    else:
        return processed_original


def avg_weather_data(df: DataFrame, mapper: DataFrame) -> DataFrame:
    """
    Compute mean aggregated weather data per county and overall.

    This function creates a deep copy of the input DataFrame, merges it
    with the county mapping, and computes the mean of all numerical
    weather features grouped by time-related columns and county, as
    well as the overall mean across all counties.

    Parameters
    ----------
    df : DataFrame
        Input weather data with latitude, longitude, datetime,
        numerical weather features.
    mapper : DataFrame
        DataFrame that maps each location to a county containing
        "county", "latitude", and "longitude" columns. Only rows with a
        known "county" (i.e., not 'unknown') are used for merging.

    Returns
    -------
    DataFrame
        DataFrame with mean aggregated weather data per county and
        overall.
    """

    # Merge weather data with county mapping, excluding 'unknown'
    # values which correspond to locations outside of Estonia
    df = pd.merge(
        left=mapper.loc[
            # (mapper["county"] != "unknown") &
            (mapper["county"] != 12),
            ["county", "latitude", "longitude"],
        ],
        right=df,
        how="left",
        on=["latitude", "longitude"],
        validate="1:m",
    )

    # Identify and exclude grouping columns from average calculation
    groups = [
        c
        for c in df.columns
        if "datetime" in c or c == "hours_ahead" or c == "data_block_id"
    ]
    excluded_c = groups + [
        "county",
        "latitude",
        "longitude",
    ]
    avg_c = [c for c in df.columns if c not in excluded_c]
    dtypes = df[avg_c].dtypes.to_dict()
    to_round = [k for k, v in dtypes.items() if np.issubdtype(v, np.integer)]

    # Compute overall mean (county='unknown'), then per-county mean,
    # and concatenate both results
    df = pd.concat(
        [
            df.groupby(groups, as_index=False, observed=True)[avg_c].mean()
            # .assign(county="unknown"),
            .assign(county=np.uint8(12)),
            df.groupby(groups + ["county"], as_index=False, observed=True)[
                avg_c
            ].mean(),
        ],
        ignore_index=True,
    )
    dtypes.update({"county": "category"})

    # Round integer-derived columns back to integers and cast all
    # columns to original dtypes
    df[to_round] = df[to_round].round()
    df = df.astype(dtypes)
    return df[["county"] + groups + avg_c]
