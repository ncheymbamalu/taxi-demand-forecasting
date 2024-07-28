"""A script that generates the one-step forecast"""

import os
import pickle

from pathlib import Path

import hopsworks
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from hopsworks.project import Project
from lightgbm import LGBMRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from src.feature_store_api import HOPSWORKS_CONFIG


def fetch_model() -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Fetches the the ML model from the 'taxi_demand_forecasting' project's Model Registry

    Returns:
        CatBoostRegressor | LGBMRegressor | XGBRegressor: Trained model
    """
    try:
        # connect to the Hopsworks 'taxi_demand_forecasting' project
        project: Project = hopsworks.login(
            project=HOPSWORKS_CONFIG.get("project"), api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )

        # load the model from the project's Model Registry
        remote_model_dir: str = (
            project
            .get_model_registry()
            .get_model(
                name=HOPSWORKS_CONFIG.get("model_registry").get("model_name"),
                version=HOPSWORKS_CONFIG.get("model_registry").get("version")
            )
            .download()
        )
        model: CatBoostRegressor | LGBMRegressor | XGBRegressor = pickle.load(
            open(Path(remote_model_dir) / "model.pkl", "rb")
        )
        return model
    except Exception as e:
        raise e


def generate_forecast(data: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    """Generates the one-step forecast for each location ID, that is, its predicted taxi 
    demand for the next hour

    Args:
        data (pd.DataFrame): Dataset containing datetime features, window features 
        (average lag features), lag features, and the target
        target (str, optional): Column name of the target variable. Defaults to "target".

    Returns:
        pd.DataFrame: Dataset containing each location ID's forecasted taxi demand
    """
    try:
        # fetch the model from the 'taxi_demand_forecasting' project's Model Registry
        model: CatBoostRegressor | LGBMRegressor | XGBRegressor = fetch_model()
        
        # specify the different types of features and the final output columns
        non_features: list[str] = ["location_id", "pickup_datetime"]
        datetime_features: list[str] = ["hour", "day_of_week"]
        lag_features: list[str] = [col for col in data.columns if col.startswith("lag")]
        window_features: list[str] = [col for col in data.columns if col.startswith("avg")]
        output_cols: list[str] = (
            non_features + datetime_features + window_features + lag_features + ["forecast"]
        )
        
        # an empty list to store each location ID's forecasted record
        # NOTE: each location ID's forecasted record will be a (1, len(output_cols)) pd.DataFrame
        forecasted_records: list[pd.DataFrame] = []
        
        # iterate over each location ID, and ...
        for location_id in tqdm(sorted(data["location_id"].unique())):
            # extract its current record, which is the last row
            x: pd.DataFrame = data.query(f"location_id == {location_id}").iloc[-1:, :]
            
            # create the forecasted record's pickup datetime
            pickup_datetime: pd.Timestamp = x.squeeze()["pickup_datetime"] + pd.Timedelta(hours=1)
            
            # create the forecasted record's datetime features
            x_datetime: list[int] = [pickup_datetime.hour, pickup_datetime.day_of_week]
            
            # create the forecasted record's lag features
            x_lag: list[float] = (
                x[lag_features + [target]]
                .drop(f"lag_{len(lag_features)}", axis=1)
                .squeeze()
                .tolist()
            )
            
            # create the forecasted record's window features, i.e., average lag features
            start: int = int(len(lag_features) / len(window_features))
            end: int = len(lag_features) + 1
            step: int = start
            x_window: list[float] = [
                np.mean(x_lag[-lag:]) for lag in reversed(range(start, end, step))
            ]
            
            # concatenate the forecasted record's input features
            # NOTE: this is the model input, and it's a (1, D) pd.DataFrame, ...
            # where D is the number of input features
            x = pd.DataFrame(
                np.stack(x_datetime + x_window + x_lag).reshape(1, -1),
                columns=datetime_features + window_features + lag_features
            )
            
            # get the forecast
            yhat: int = max(0, np.round(model.predict(x)[0]))
            
            # add the location ID, pickup datetime, and forecast to the input features 
            x = (
                x
                .assign(
                    location_id=location_id,
                    pickup_datetime=pickup_datetime,
                    forecast=yhat
                )
                [output_cols]
            )
            
            # append this location ID's forecasted record to the 'forecasted_records' list
            forecasted_records.append(x)
        return pd.concat(forecasted_records, axis=0, ignore_index=True).sort_values("location_id")
    except Exception as e:
        raise e
