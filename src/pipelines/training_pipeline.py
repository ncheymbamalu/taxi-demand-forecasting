"""This module provides functionality to evaluate the pipeline's current ML model."""

import pandas as pd

from src.feature_store_api import HOPSWORKS_CONFIG, get_project
from src.inference import generate_forecast
from src.logger import logger
from src.train import NaiveForecast, compute_metrics, upload_model
from src.transform import fetch_and_transform


def evaluate_model() -> None:
    """Evaluates the current forecasting model on the latest NYC taxi demand
    data and updates/replaces it if necessary
    """
    try:
        # fetch the latest validated and pre-processed data from Hopsworks, and ...
        # transform it into machine learning-ready features and labels
        data: pd.DataFrame = fetch_and_transform()

        # a dictionary that maps each location ID to its latest pickup time
        latest_pickup_times: dict[int, pd.Timestamp] = {
            location_id: data.query(f"location_id == {location_id}")["pickup_time"].max()
            for location_id in sorted(data["location_id"].unique())
        }

        # create the forecasted records
        # NOTE: the 'forecast_data' pd.DataFrame contains each location ID's predicted taxi ...
        # demand for the current hour
        records: list[pd.DataFrame] = [
            data.query(f"location_id == {location_id} & pickup_time < '{latest_pickup_time}'")
            for location_id, latest_pickup_time in latest_pickup_times.items()
        ]
        forecast_data: pd.DataFrame = generate_forecast(
            pd.concat(records, axis=0, ignore_index=True)
        )

        # extract the actual records
        # NOTE: the 'data' pd.DataFrame contains each location ID's actual number of taxi rides ...
        # for the current hour
        records = [
            data.query(f"location_id == {location_id} & pickup_time == '{latest_pickup_time}'")
            for location_id, latest_pickup_time in latest_pickup_times.items()
        ]
        data = pd.concat(records, axis=0, ignore_index=True)

        # merge the forecasted records with the actual records
        data = (
            data[["location_id", "target", "lag_1"]]
            .merge(forecast_data[["location_id", "forecast"]], how="inner", on="location_id")
            .drop("location_id", axis=1)
        )

        # get the RMSE for the naive forecast and one-step forecast
        naive: float = compute_metrics(data["target"], NaiveForecast().predict(data)).get("rmse")
        one_step: float = compute_metrics(data["target"], data["forecast"]).get("rmse")

        # if the one-step forecast is worse than the naive forecast, then ...
        if one_step > naive:
            # delete the current model, its associated files, and metadata from Hopsworks
            logger.info("The current forecasting model is unsatisfactory and will be replaced.")
            (
                get_project()
                .get_model_registry()
                .get_model(
                    name=HOPSWORKS_CONFIG.model_registry.model_name,
                    version=HOPSWORKS_CONFIG.model_registry.version
                )
                .delete()
            )

            # train and evaluate a new model and upload it to Hopsworks
            upload_model()
        else:
            logger.info("The current forecasting model is fine.")
    except Exception as e:
        raise e


if __name__ == "__main__":
    evaluate_model()
