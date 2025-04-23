import pandas as pd


def load_raw_data(path: str) -> dict[str, pd.DataFrame | pd.Series]:
    """Load all data and return dict with DFs."""
    return {
        "train": pd.read_csv(f"{path}train.csv"),
        "gas_prices": pd.read_csv(f"{path}gas_prices.csv"),
        "client": pd.read_csv(f"{path}client.csv"),
        "electricity_prices": pd.read_csv(f"{path}electricity_prices.csv"),
        "forecast_weather": pd.read_csv(f"{path}forecast_weather.csv"),
        "historical_weather": pd.read_csv(f"{path}historical_weather.csv"),
        "station_county_mapping": pd.read_csv(
            f"{path}weather_station_to_county_mapping.csv"
        ),
        "county_id_to_name_map": pd.read_json(
            f"{path}county_id_to_name_map.json", typ="series"
        ).str.lower(),
    }
