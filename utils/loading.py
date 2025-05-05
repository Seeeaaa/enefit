import pandas as pd
from pandas import DataFrame, Series


def load_all_raw_data(
    main_path: str, additional_path: str | None = None
) -> tuple[
    dict[str, DataFrame | Series], dict[str, DataFrame | Series] | None
]:
    """
    Load all raw datasets from competition and external sources.

    Parameters
    ----------
    main_path : str
        Directory containing competition CSV and JSON files.
    additional_path : str | None = None
        Directory containing external datasets. If None, additional data
        is not loaded.

    Returns
    -------
    tuple[
        dict[str, DataFrame | Series],
        dict[str, DataFrame | Series] | None
    ]
        Tuple containing two dictionaries:
        - Dictionary with the original datasets. Keys are:
        'train', 'gas_prices', 'client', 'electricity_prices',
        'forecast_weather', 'historical_weather',
        'station_county_mapping', 'county_id_to_name_map'.
        - Dictionary with the additional datasets. Keys are:
          - 'holidays' (if additional_path is not None)
    """
    original_data = {
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
        "station_county_mapping": pd.read_csv(
            f"{main_path}weather_station_to_county_mapping.csv"
        ),
        "county_id_to_name_map": pd.read_json(
            f"{main_path}county_id_to_name_map.json", typ="series"
        ),
    }
    additional_data = None
    if additional_path:
        additional_data = {
            "holidays": pd.read_csv(f"{additional_path}holidays.csv")
        }
    return original_data, additional_data
