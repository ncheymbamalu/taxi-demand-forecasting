"""A script that trains and evaluates a machine learning model and uploads it to Hopsworks"""

import os
import pickle

from pathlib import PosixPath

import hopsworks
import pandas as pd
import polars as pl

from hopsworks.project import Project
from hsfs.feature_group import FeatureGroup
from hsml.model_registry import ModelRegistry
from hsml.model_schema import ModelSchema, Schema
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import Config
from src.feature_store_api import HOPSWORKS_CONFIG, get_feature_group
from src.logger import logging
from src.train import NaiveForecast, compute_metrics, split_data, train_model
from src.transform import tabularize_data


def main() -> None:
    try:
        # load the pre-processed data from Hopsworks
        feature_group: FeatureGroup = get_feature_group()
        data: pd.DataFrame = feature_group.read()
        if not data.empty:
            # convert the pre-processed data to tabular, ML-ready data
            data = (
                pl.from_pandas(data)
                .with_columns(pl.from_epoch(pl.col("unix_epoch_ms"), time_unit="ms"))
                .rename({"unix_epoch_ms": "pickup_datetime"})
                .sort(["location_id", "pickup_datetime"])
                .to_pandas()
                .pipe(tabularize_data)
            )

            # split the ML-ready data into train and test sets
            x_train, y_train, x_test, y_test = split_data(data)

            # extract the 'baseline' metrics
            baseline_metrics: dict[str, float] = compute_metrics(
                y_test, NaiveForecast().predict(x_test)
            )

            # train and evaluate the model
            model: XGBRegressor | LGBMRegressor = train_model(data)
            (
                model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
                if isinstance(model, XGBRegressor)
                else model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
            )
            metrics: dict[str, float] = compute_metrics(y_test, model.predict(x_test))

            # if the model's predictions are better than the naive forecast, then ...
            if metrics.get("rmse") < baseline_metrics.get("rmse"):
                # save the model locally as ~/artifacts/model.pkl
                output_dir: PosixPath = Config.ARTIFACTS_DIR
                output_dir.mkdir(parents=True, exist_ok=True)
                pickle.dump(model, open(output_dir / "model.pkl", "wb"))

                # connect to the Hopsworks 'taxi_demand_forecasting' project
                project: Project = hopsworks.login(
                    project=HOPSWORKS_CONFIG.get("project"),
                    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
                )

                # create an object that points to the project's model registry
                model_registry: ModelRegistry = project.get_model_registry()

                # upload ~/artifacts/model.pkl to the project's model registry
                (
                    model_registry.sklearn.create_model(
                        name="one_step_forecaster",
                        metrics=metrics,
                        description=model.__str__(),
                        input_example=x_train.sample(),
                        model_schema=ModelSchema(
                            input_schema=Schema(x_train), output_schema=Schema(y_train)
                        ),
                    ).save(os.path.join(output_dir, "model.pkl"))
                )
            else:
                logging.info(
                    "Predictions are worse than the naive forecast, so the model will not be saved."
                )
        else:
            logging.info(
                "No data exists for the '%s' project's '%s' feature group",
                feature_group._get_project_name(),
                feature_group.name,
            )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
