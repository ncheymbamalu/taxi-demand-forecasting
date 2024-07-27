"""A script that trains and evaluates select ML models and uploads the 'best' one to Hopsworks"""

import os
import pickle

from pathlib import PosixPath

import hopsworks
import pandas as pd
import polars as pl

from catboost import CatBoostRegressor
from dotenv import load_dotenv
from hsml.model_registry import ModelRegistry
from hsml.model_schema import ModelSchema, Schema
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import Config
from src.feature_store_api import HOPSWORKS_CONFIG
from src.inference import generate_forecast
from src.logger import logging
from src.train import compute_metrics, split_data, train_model
from src.transform import fetch_and_transform, tabularize_data

load_dotenv(Config.HOME_DIR / ".env")


def upload_model() -> None:
    """Trains several ML models on the latest hourly NYC taxi demand data, selects 
    the 'best' one, and uploads it to the Hopsworks project's Model Registry
    """
    try:
        # fetch the latest taxi demand data from Hopsworks project's Feature Store/Group
        data: pd.DataFrame = (
            (
                hopsworks
                .login(
                    project=HOPSWORKS_CONFIG.get("project"),
                    api_key_value=os.getenv("HOPSWORKS_API_KEY")
                )
                .get_feature_store()
                .get_or_create_feature_group(**HOPSWORKS_CONFIG.get("feature_group"))
                .read()
            )
        )
        if data.empty:
            logging.info(
                "No data exists for the '%s' project's '%s' feature group",
                HOPSWORKS_CONFIG.get("project"), # feature_group._get_project_name()
                HOPSWORKS_CONFIG.get("feature_group").get("name"), # feature_group.name
            )
        else:
            # transform the taxi demand data into features and labels
            data = (
                pl.from_pandas(data)
                .with_columns(pl.from_epoch(pl.col("unix_epoch_ms"), time_unit="ms"))
                .rename({"unix_epoch_ms": "pickup_datetime"})
                .sort(["location_id", "pickup_datetime"])
                .to_pandas()
                .pipe(tabularize_data)
            )
            
            # train and evaluate several models and extract the 'best' one
            model: CatBoostRegressor | LGBMRegressor | XGBRegressor = train_model(data)

            # get the test set feature matrix and target vector
            _, _, x_test, y_test = split_data(data)

            # compute the model's test set metrics (RMSE and R²)
            metrics: dict[str, float] = compute_metrics(y_test, model.predict(x_test))
            
            # save the model locally as ~/artifacts/model.pkl
            output_dir: PosixPath = Config.ARTIFACTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            pickle.dump(model, open(output_dir / "model.pkl", "wb"))

            # connect to the Hopsworks project's Model Registry 
            model_registry: ModelRegistry = (
                hopsworks
                .login(
                    project=HOPSWORKS_CONFIG.get("project"),
                    api_key_value=os.getenv("HOPSWORKS_API_KEY")
                )
                .get_model_registry()
            )

            # upload ~/artifacts/model.pkl to the Model Registry
            logging.info(
                "Uploading the %s to the '%s' project's Model Registry under the name, '%s'",
                model.__class__.__name__,
                HOPSWORKS_CONFIG.get("project"),
                HOPSWORKS_CONFIG.get("model_registry").get("model_name")
            )
            (
                model_registry.sklearn
                .create_model(
                    name=HOPSWORKS_CONFIG.get("model_registry").get("model_name"),
                    version=HOPSWORKS_CONFIG.get("model_registry").get("version"),
                    metrics=metrics,
                    description=model.__str__(),
                    input_example=x_test.sample(),
                    model_schema=ModelSchema(
                        input_schema=Schema(x_test), output_schema=Schema(y_test)
                    ),
                )
                .save(os.path.join(output_dir, "model.pkl"))
            )
    except Exception as e:
        raise e
    

def evaluate_model() -> None:
    """Evaluates the Hopsworks project's current model on the latest hourly 
    NYC taxi demand data, and updates/replaces it if necessary"""
    try:
        # fetch the latest taxi demand data from the Hopsworks Feature Store/Group, and ...
        # transform it into features and labels
        data: pd.DataFrame = fetch_and_transform()
        
        # a dictionary that maps each location ID to its latest timestamp
        latest_timestamps: dict[str, pd.Timestamp] = {
            location_id: data.query(f"location_id == {location_id}")["pickup_datetime"].max()
            for location_id in sorted(data["location_id"].unique())
        }
        
        # create the forecasted records
        # NOTE: the 'forecast_data' pd.DataFrame contains each location ID's forecast, i.e., ...
        # predicted taxi demand for the current hour
        data_to_forecast: list[pd.DataFrame] = [
            data.query(f"location_id == {location_id} & pickup_datetime < '{pickup_datetime}'")
            for location_id, pickup_datetime in latest_timestamps.items()
        ]
        forecast_data: pd.DataFrame = generate_forecast(
            pd.concat(data_to_forecast, axis=0, ignore_index=True)
        )
        
        # extract the actual records
        # NOTE: this pd.DataFrame contains each location ID's actual taxi rides, ...
        # for the current hour
        actual_records: list[pd.DataFrame] = [
            data.query(f"location_id == {location_id} & pickup_datetime == '{pickup_datetime}'")
            for location_id, pickup_datetime in latest_timestamps.items()
        ]
        data = pd.concat(actual_records, axis=0, ignore_index=True)
        
        # merge the forecasted records with the actual records
        data = (
            forecast_data[["location_id", "forecast"]]
            .merge(data[["location_id", "lag_1", "target"]], how="inner", on="location_id")
            .drop("location_id", axis=1)
        )
        
        # compute the RMSE for both the naive forecast and one-step forecast
        naive_rmse: float = compute_metrics(data["target"], data["lag_1"]).get("rmse")
        one_step_rmse: float = compute_metrics(data["target"], data["forecast"]).get("rmse")
        
        # if the one-step forecast is worse than the naive forecast, then ...
        if one_step_rmse > naive_rmse:
            # delete the current model, its associated files, and metadata ...
            # from the Hopsworks project's Model Registry, and ...
            logging.info("The current forecasting model is unsatisfactory and will be replaced.")
            (
                hopsworks
                .login(
                    project=HOPSWORKS_CONFIG.get("project"),
                    api_key_value=os.getenv("HOPSWORKS_API_KEY")
                )
                .get_model_registry()
                .get_model(
                    name=HOPSWORKS_CONFIG.get("model_registry").get("model_name"), 
                    version=HOPSWORKS_CONFIG.get("model_registry").get("version")
                )
                .delete()
            )
            
            # train a new model and upload it to the project's Model Registry
            upload_model()
        else:
            logging.info("The current forecasting model is fine.")
    except Exception as e:
        raise e


if __name__ == "__main__":
    evaluate_model()
