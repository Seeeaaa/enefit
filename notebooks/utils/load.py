import pandas as pd
from pandas import DataFrame, Series


def load_all_raw_data(
    main_path: str, additional_path: str
) -> tuple[dict[str, DataFrame | Series], dict[str, DataFrame]]:
    """
    Load all raw datasets from competition and external sources.

    Parameters
    ----------
    main_path : str
        Directory containing competition CSV and JSON files.
    additional_path : str
        Directory containing external datasets.

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series]
        Dictionary with the following keys:
          - 'train'
          - 'gas_prices'
          - 'client'
          - 'electricity_prices'
          - 'forecast_weather'
          - 'historical_weather'
          - 'station_county_mapping'
          - 'county_id_to_name_map'
          - 'holidays'
    """
    return {
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
    }, {
        "holidays": pd.read_csv(f"{additional_path}holidays.csv"),
    }
