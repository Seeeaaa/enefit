import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from utils.preprocessing import avg_weather_data
from utils.feature_engineering import (
    get_lag,
    add_lags,
    prepare_time_series_groupby,
    compute_rolling_features,
    add_dst_flag,
    add_cyclic_datetime_features,
)


class DataPipeline:
    def __init__(
        self,
        parquet_path: str,
    ):
        self.parquet_path = parquet_path
        self._raw: dict[str, DataFrame] = {}
        self._prepared: dict[str, DataFrame] = {}
        self.df: DataFrame = pd.DataFrame()

    def load(self) -> None:
        """
        Load raw datasets from parquet files into self.raw.
        """
        self._raw = {
            "train": pd.read_parquet(f"{self.parquet_path}train"),
            "gas_prices": pd.read_parquet(f"{self.parquet_path}gas_prices"),
            "client": pd.read_parquet(f"{self.parquet_path}client"),
            "electricity_prices": pd.read_parquet(
                f"{self.parquet_path}electricity_prices"
            ),
            "forecast_weather": pd.read_parquet(
                f"{self.parquet_path}forecast_weather"
            ),
            "historical_weather": pd.read_parquet(
                f"{self.parquet_path}historical_weather"
            ),
            "weather_station_to_county_mapping": pd.read_parquet(
                f"{self.parquet_path}weather_station_to_county_mapping"
            ),
            "holidays": pd.read_parquet(f"{self.parquet_path}holidays"),
        }

    def _prepare_train(self) -> None:
        df = self._raw["train"].copy()
        df = (
            df[
                [
                    "county",
                    "product_type",
                    "is_business",
                    "is_consumption",
                    "datetime",
                    "target",
                    "data_block_id",
                ]
            ]
            .dropna()
            .astype(
                {
                    "county": "category",
                    "product_type": "category",
                    # "is_business": "bool",
                    # "is_consumption": "bool",
                    "is_business": "uint8",
                    "is_consumption": "uint8",
                    "datetime": "datetime64[ns]",
                    "target": "float32",
                    "data_block_id": "uint16",
                }
            )
            .astype(
                {
                    "is_business": "category",
                    "is_consumption": "category",
                }
            )
        )
        df["date"] = df["datetime"].dt.normalize()
        self._prepared["train"] = df

    def _prepare_gas_prices(self) -> None:
        df = self._raw["gas_prices"].copy()
        df = df[
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
        self._prepared["gas_prices"] = df

    def _prepare_client(self) -> None:
        df = self._raw["client"].copy()
        df = (
            df[
                [
                    "county",
                    "product_type",
                    "is_business",
                    "eic_count",
                    "installed_capacity",
                    "data_block_id",
                ]
            ]
            .astype(
                {
                    "county": "category",
                    "product_type": "category",
                    # "is_business": "bool",
                    "is_business": "uint8",
                    "eic_count": "float32",
                    "installed_capacity": "float32",
                    "data_block_id": "uint16",
                }
            )
            .astype({"is_business": "category"})
        )
        self._prepared["client"] = df

    def _prepare_electricity_prices(self) -> None:
        df = self._raw["electricity_prices"].copy()
        df = df.astype(
            {
                "origin_date": "datetime64[ns]",
                "euros_per_mwh": "float32",
                "data_block_id": "uint16",
            }
        )
        df["datetime"] = df["origin_date"] + pd.Timedelta(2, "d")
        df = df[["datetime", "euros_per_mwh", "data_block_id"]]
        self._prepared["electricity_prices"] = df

    def _prepare_forecast_weather(self) -> None:
        df = self._raw["forecast_weather"].copy()
        precipitation_threshold = 0.1  # Threshold in mm
        df[["latitude", "longitude"]] = (
            df[["latitude", "longitude"]].round(1).mul(10)
        )
        df["hours_ahead"] = pd.to_timedelta(df["hours_ahead"], "h")
        df["snowfall_mm"] = df["snowfall"].mul(1000)
        df["total_precipitation_mm"] = df["total_precipitation"].mul(1000)

        cols = ["snowfall_mm", "total_precipitation_mm"]
        df[cols] = df[cols].where(df[cols].abs() >= precipitation_threshold, 0)
        df = df.rename(
            columns={
                "10_metre_u_wind_component": "u_component",
                "10_metre_v_wind_component": "v_component",
            }
        )
        df["windspeed"] = np.sqrt(
            df["u_component"] ** 2 + df["v_component"] ** 2
        )
        df[["direct_solar_radiation", "surface_solar_radiation_downwards"]] = (
            df[
                ["direct_solar_radiation", "surface_solar_radiation_downwards"]
            ].clip(lower=0)
        )
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
                "cloudcover_low": "float32",
                "cloudcover_mid": "float32",
                "cloudcover_high": "float32",
                "cloudcover_total": "float32",
                "u_component": "float32",
                "v_component": "float32",
                "windspeed": "float32",
                "direct_solar_radiation": "float32",
                "surface_solar_radiation_downwards": "float32",
            }
        )
        self._prepared["forecast_weather"] = df

    def _prepare_historical_weather(self) -> None:
        df = self._raw["historical_weather"].copy()
        hw_to_drop = [1176339, 1176343]  # Drop duplicates
        df = df.drop(index=hw_to_drop).reset_index(drop=True)
        df[["latitude", "longitude"]] = (
            df[["latitude", "longitude"]].round(1).mul(10)
        )
        snow_water_density_ratio = 0.1
        df["snowfall"] = df["snowfall"].mul(10).mul(snow_water_density_ratio)
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
                "cloudcover_low": "float32",
                "cloudcover_mid": "float32",
                "cloudcover_high": "float32",
                "cloudcover_total": "float32",
                "windspeed": "float32",
                "u_component": "float32",
                "v_component": "float32",
                "shortwave_radiation": "float32",
                "direct_solar_radiation": "float32",
                "diffuse_radiation": "float32",
            }
        )
        self._prepared["historical_weather"] = df

    def _prepare_weather_station_to_county_mapping(self) -> None:
        df = self._raw["weather_station_to_county_mapping"].copy()
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
        df = (
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
        self._prepared["weather_station_to_county_mapping"] = df

    def _prepare_holidays(self) -> None:
        df = self._raw["holidays"].copy()
        df = df.drop(columns=["name"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["value"] = True
        df = df.pivot_table(
            index="date",
            columns="holiday_type",
            values="value",
            aggfunc="any",
            fill_value=False,
        ).reset_index()
        df.columns.name = None
        self._prepared["holidays"] = df

    def prepare(self, drop_raw=True) -> None:
        self._prepare_train()
        self._prepare_gas_prices()
        self._prepare_client()
        self._prepare_electricity_prices()
        self._prepare_forecast_weather()
        self._prepare_historical_weather()
        self._prepare_weather_station_to_county_mapping()
        self._prepare_holidays()
        if drop_raw:
            self._raw.clear()

    def merge(self, drop_prepared=True) -> None:
        fp = "f1_"  # Prefix for columns related to the 1 day forecast
        hp = "h2_"  # Prefix for columns related to 2 day historical data
        holidays_names = self._prepared["holidays"].columns.drop(["date"])
        self.df = (
            pd.merge(
                left=self._prepared["train"],
                right=self._prepared["client"],
                how="left",
                on=["county", "product_type", "is_business", "data_block_id"],
                validate="m:1",
            )
            .merge(
                right=self._prepared["gas_prices"],
                how="left",
                on=["data_block_id"],
                validate="m:1",
            )
            .merge(
                right=self._prepared["electricity_prices"],
                how="left",
                on=["datetime", "data_block_id"],
                validate="m:1",
            )
            .merge(
                avg_weather_data(
                    self._prepared["forecast_weather"],
                    self._prepared["weather_station_to_county_mapping"],
                )
                .drop_duplicates(
                    ["county", "forecast_datetime", "data_block_id"],
                    keep="last",
                )
                .add_prefix(fp),
                how="left",
                left_on=["county", "datetime", "data_block_id"],
                right_on=[
                    fp + c
                    for c in ["county", "forecast_datetime", "data_block_id"]
                ],
                validate="m:1",
            )
            .drop(
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
            .merge(
                avg_weather_data(
                    self._prepared["historical_weather"],
                    self._prepared["weather_station_to_county_mapping"],
                )
                .assign(
                    fully_available_at=lambda x: x["datetime"]
                    + pd.Timedelta("2 d")
                )
                .add_prefix(hp),
                how="left",
                left_on=["county", "datetime"],
                right_on=[hp + c for c in ["county", "fully_available_at"]],
                validate="m:1",
            )
            .drop(
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
            .merge(
                right=self._prepared["holidays"],
                how="left",
                on="date",
            )
            .fillna({column: False for column in holidays_names})
            # .astype({column: "bool" for column in holidays_names})
            .astype({column: "uint8" for column in holidays_names})
            .astype({column: "category" for column in holidays_names})
        )

        if drop_prepared:
            self._prepared.clear()
        self.df = self.df.drop(columns=["data_block_id", "date"])

    def add_features(
        self,
        lag_specs: dict[str, dict[str, list[str]]],
        group_cols: list,
        datetime_col: str,
        value_col: str,
    ) -> None:

        self.df = (
            add_dst_flag(self.df)
            .astype({"dst": "uint8"})
            .astype({"dst": "category"})
        )
        self.df = add_cyclic_datetime_features(self.df, drop_raw=True)
        self.df = add_lags(
            self.df,
            value_col,
            list(lag_specs.keys()),
            datetime_col,
            group_cols,
        )
        for lag, windows_params in lag_specs.items():
            tsg = prepare_time_series_groupby(
                self.df,
                f"target_lag_{lag}",
                datetime_col,
                group_cols,
            )
            for window, funcs in windows_params.items():
                self.df = self.df.merge(
                    compute_rolling_features(
                        tsg,
                        f"target_lag_{lag}",
                        window,
                        funcs,
                    ),
                    "left",
                    group_cols + [datetime_col],
                    validate="1:1",
                )
        self.df["target_to_capacity"] = self.df["target_lag_2d"] / (
            self.df["installed_capacity"]
        )
        self.df["capacity_to_eic"] = self.df["installed_capacity"] / (
            self.df["eic_count"]
        )
