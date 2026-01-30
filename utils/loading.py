import pandas as pd
from pandas import DataFrame, Series


def load_raw_data(
    main_path: str, additional_path: str | None = None
) -> dict[str, DataFrame]:
    """
    Load all raw datasets from competition and optional external
    sources.

    Parameters
    ----------
    main_path : str
        Path to the directory containing competition CSV and JSON
        files.
    additional_path : str | None, optional
        Path to the directory containing external datasets. If None, no
        additional data is loaded.

    Returns
    -------
    dict[str, DataFrame]
        A dictionary where keys are dataset names (e.g., 'train',
        'client', 'holidays'), and values are either DataFrame or
        Series.
    """
    data = {
        "train": pd.read_csv(f"{main_path}train.csv"),
        "gas_prices": pd.read_csv(f"{main_path}gas_prices.csv"),
        "client": pd.read_csv(f"{main_path}client.csv"),
        "electricity_prices": pd.read_csv(
            f"{main_path}electricity_prices.csv"
        ),
        "forecast_weather": pd.read_csv(f"{main_path}forecast_weather.csv"),
        "historical_weather": pd.read_csv(
            f"{main_path}historical_weather.csv"
        ),
        "weather_station_to_county_mapping": pd.read_csv(
            f"{main_path}weather_station_to_county_mapping.csv"
        ),
        "county_id_to_name_map": pd.read_json(
            f"{main_path}county_id_to_name_map.json", typ="series"
        ),
    }
    if additional_path:
        data["holidays"] = pd.read_csv(f"{additional_path}holidays.csv")

    return data


def load_parquet_data(parquet_path: str) -> dict[str, DataFrame]:
    """
    Load raw data in parquet format.

    Returns
    -------
    dict[str, DataFrame]
        A dictionary where keys are dataset names (e.g., 'train'), and
        values are DataFrames.
    """
    data = {
        "train": pd.read_parquet(f"{parquet_path}train"),
        "gas_prices": pd.read_parquet(f"{parquet_path}gas_prices"),
        "client": pd.read_parquet(f"{parquet_path}client"),
        "electricity_prices": pd.read_parquet(
            f"{parquet_path}electricity_prices"
        ),
        "forecast_weather": pd.read_parquet(f"{parquet_path}forecast_weather"),
        "historical_weather": pd.read_parquet(
            f"{parquet_path}historical_weather"
        ),
        "weather_station_to_county_mapping": pd.read_parquet(
            f"{parquet_path}weather_station_to_county_mapping"
        ),
        "holidays": pd.read_parquet(f"{parquet_path}holidays"),
    }

    return data
