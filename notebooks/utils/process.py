import pandas as pd


def process_train(df: pd.DataFrame) -> pd.DataFrame:
    return df[
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
            "target": "float32",
            "data_block_id": "uint16",
            "datetime": "datetime64[ns]",
        }
    )


def process_gas(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["origin_date", "forecast_date"]).astype(
        {
            "data_block_id": "uint16",
            "lowest_price_per_mwh": "float32",
            "highest_price_per_mwh": "float32",
        }
    )[
        [
            "data_block_id",
            "lowest_price_per_mwh",
            "highest_price_per_mwh",
        ]
    ]


def process_client(df: pd.DataFrame) -> pd.DataFrame:
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
            "date": "datetime64[ns]",
            "eic_count": "uint32",
            "installed_capacity": "float32",
            "data_block_id": "uint16",
        }
    )


def process_electricity(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "origin_date": "datetime64[ns]",
            "euros_per_mwh": "float32",
            "data_block_id": "uint16",
        }
    ).assign(
        electricity_datetime=lambda x: x["origin_date"] + pd.Timedelta(2, "d")
    )[["electricity_datetime", "euros_per_mwh", "data_block_id"]]


def process_fweather(df: pd.DataFrame) -> pd.DataFrame:
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
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "cloudcover_total",
            "10_metre_u_wind_component",
            "10_metre_v_wind_component",
            "direct_solar_radiation",
            "surface_solar_radiation_downwards",
            "snowfall",
            "total_precipitation",
        ]
    ]

    df[["latitude", "longitude"]] = (
        df[["latitude", "longitude"]].round(1).mul(10)
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

    df = df.astype(
        {
            "latitude": "uint16",
            "longitude": "uint16",
            "origin_datetime": "datetime64[ns]",
            "forecast_datetime": "datetime64[ns]",
            "data_block_id": "uint16",
            "temperature": "float32",
            "dewpoint": "float32",
            "cloudcover_low": "uint8",
            "cloudcover_mid": "uint8",
            "cloudcover_high": "uint8",
            "cloudcover_total": "uint8",
            "10_metre_u_wind_component": "float32",
            "10_metre_v_wind_component": "float32",
            "direct_solar_radiation": "float32",
            "surface_solar_radiation_downwards": "float32",
            "snowfall": "float32",
            "total_precipitation": "float32",
        }
    )
    df["hours_ahead"] = pd.to_timedelta(df["hours_ahead"], "h")
    return df


def process_hweather(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        [
            "latitude",
            "longitude",
            "datetime",
            "data_block_id",
            "temperature",
            "dewpoint",
            "rain",
            "snowfall",
            "surface_pressure",
            "cloudcover_total",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "windspeed_10m",
            "winddirection_10m",
            "shortwave_radiation",
            "direct_solar_radiation",
            "diffuse_radiation",
        ]
    ]

    df[["latitude", "longitude"]] = (
        df[["latitude", "longitude"]].round(1).mul(10)
    )
    df = df.astype(
        {
            "latitude": "uint16",
            "longitude": "uint16",
            "datetime": "datetime64[ns]",
            "data_block_id": "uint16",
            "temperature": "float32",
            "dewpoint": "float32",
            "rain": "float32",
            "snowfall": "float32",
            "surface_pressure": "float32",
            "cloudcover_total": "uint8",
            "cloudcover_low": "uint8",
            "cloudcover_mid": "uint8",
            "cloudcover_high": "uint8",
            "windspeed_10m": "float32",
            "winddirection_10m": "uint16",
            "shortwave_radiation": "uint16",
            "direct_solar_radiation": "uint16",
            "diffuse_radiation": "uint16",
        }
    )
    hw_to_drop = [1176339, 1176343]
    df = df.drop(index=hw_to_drop).reset_index(drop=True)
    return df


def process_stations(df: pd.DataFrame) -> pd.DataFrame:
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
        .rename(
            columns={
                "county_name": "county",
                "county": "county_index",
            }
        )
    )
