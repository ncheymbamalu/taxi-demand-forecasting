"""A script that generates the one-step forecast"""

import pickle

from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from src.feature_store_api import HOPSWORKS_CONFIG, get_project


def fetch_model() -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Fetches the ML model from the Hopsworks Model Registry

    Returns:
        CatBoostRegressor | LGBMRegressor | XGBRegressor: Trained model
    """
    try:
        # path to the remote directory where the model artifact is stored
        remote_model_dir: str = (
            get_project()
            .get_model_registry()
            .get_model(
                name=HOPSWORKS_CONFIG.model_registry.model_name,
                version=HOPSWORKS_CONFIG.model_registry.version
            )
            .download()
        )
        return pickle.load(open(Path(remote_model_dir) / "model.pkl", "rb"))
    except Exception as e:
        raise e


def generate_forecast(data: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    """Returns a pd.DataFrame that contains each location ID's one-step forecast,
    that is, its predicted taxi demand for the upcoming hour

    Args:
        data (pd.DataFrame): Machine learning-ready data consisting of datetime features,
        window features (average lag features), lag features, and the target
        target (str, optional): Column name of the target variable. Defaults to "target".

    Returns:
        pd.DataFrame: Dataset containing each location ID's forecasted taxi demand
    """
    try:
        # fetch the model from Hopsworks
        model: CatBoostRegressor | LGBMRegressor | XGBRegressor = fetch_model()

        # specify the different types of features and final output columns
        non_features: list[str] = ["location_id", "pickup_time"]
        temporal_features: list[str] = ["day_of_week", "hour"]
        window_features: list[str] = [col for col in data.columns if col.startswith("avg")]
        lag_features: list[str] = [col for col in data.columns if col.startswith("lag")]
        output_cols: list[str] = (
            non_features + temporal_features + window_features + lag_features + ["forecast"]
        )

        # an empty list to store each location IDs forecasted record
        # NOTE: each location ID's forecasted record with be a (1, len(output_cols)) pd.DataFrame
        forecasted_records: list[pd.DataFrame] = []

        # iterate over each location ID, and ...
        for location_id in tqdm(sorted(data["location_id"].unique())):
            # extract the current record, which is the last row
            x: pd.DataFrame = data.query(f"location_id == {location_id}").iloc[-1:, :]

            # create the forecasted record's pickup time
            pickup_time: pd.Timestamp = x.squeeze()["pickup_time"] + pd.Timedelta(hours=1)

            # create the forecasted record's temporal features
            x_temporal: list[int] = [pickup_time.day_of_week, pickup_time.hour]

            # create the forecasted record's lag features
            x_lag: list[float] = x[lag_features[1:] + [target]].squeeze().tolist()

            # create the forecast record's window features, that is, average lag features
            start = step = 4
            end: int = len(lag_features) + 1
            x_window: list[float] = [
                np.mean(x_lag[-lag:]) for lag in reversed(range(start, end, step))
            ]

            # concatenate the forecasted record's input features
            # NOTE: this is the model input, and it's a (1, D) pd.DataFrame, ...
            # where D is the number of input features
            x: pd.DataFrame = pd.DataFrame(
                data=[x_temporal + x_window + x_lag],
                columns=temporal_features + window_features + lag_features
            )

            # generate the one-step forecast
            forecast: int = max(0, int(round(model.predict(x)[0])))

            # add the location ID, pickup datetime, and forecast to the input features
            x = (
                x.assign(location_id=location_id, pickup_time=pickup_time, forecast=forecast)
                [output_cols]
            )

            # append this location ID's forecasted record to the 'forecasted_records' list
            forecasted_records.append(x)
        return (
            pd.concat(forecasted_records, axis=0)
            .sort_values(by="location_id")
            .reset_index(drop=True)
        )
    except Exception as e:
        raise e
