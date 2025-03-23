"""This module contains utility/helper functions."""

import os
import pickle

from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import PosixPath
from typing import Any

import httpx
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl

from httpx import Response
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import Apply
from omegaconf import OmegaConf
from plotly.graph_objects import Figure
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
from xgboost import XGBRegressor

from src.config import Paths, data_config, model_config
from src.logger import logger


# --------------------------------------------------------------------------------------------------
# Data utility functions
# --------------------------------------------------------------------------------------------------
@logger.catch
def fetch_data(year: int, month: int) -> pl.DataFrame:
    """Fetches raw data from the NYC taxi data API, then pre-processes, validates, and
    returns it as a pl.DatFrame.

    Args:
        year (int): Raw data's recorded year.
        month (int): Raw data's recorded month.

    Returns:
        pl.DataFrame: Pre-processed and validated data that contains hourly NYC taxi rides.
    """
    try:
        path: str = os.path.join(Paths.TAXI_DATA_API, f"yellow_tripdata_{year}-{month:02d}.parquet")
        response: Response = httpx.get(url=path)
        if response.status_code == 200:
            logger.info(f"Downloading, pre-processing, and validating raw data from {path}.")
            data: pl.DataFrame = pl.read_parquet(path)
            dfs: list[pl.DataFrame] = [
                (
                    data
                    .filter(pl.col("PULocationID").eq(location_id))
                    .sort(by="tpep_pickup_datetime")
                    .with_columns(
                        pl.col("tpep_pickup_datetime").dt.truncate("1h")
                    )
                    .group_by("tpep_pickup_datetime")
                    .count()
                    .upsample(
                        time_column="tpep_pickup_datetime",
                        every="1h",
                        maintain_order=True
                    )
                    .fill_null(0)
                    .with_columns(
                        pl.col("count").cast(pl.Int32),
                        location_id=location_id
                    )
                    .rename({
                        "tpep_pickup_datetime": "pickup_time",
                        "count": "n_rides"
                    })
                    .select(data_config.columns)
                )
                for location_id in tqdm(sorted(data["PULocationID"].unique()), unit="Location ID")
            ]
            data = (
                pl.concat(dfs, how="vertical")
                .unique(maintain_order=True, keep="first")
            )
            assert data.is_duplicated().sum() == 0
            assert data.null_count().sum_horizontal()[0] == 0
            return data
        logger.info(f"Invalid request. {path} is not available to download.")
        return pl.DataFrame(schema=data_config.columns)
    except Exception as e:
        raise e


@logger.catch
def transform_data(
    data: pl.DataFrame,
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column,
    max_lag: int = 24
) -> pl.DataFrame:
    """Transforms pre-processed and validated hourly taxi rides data into an ML-ready
    dataset that contains lag features, average lag features, datetime features, and
    the corresponding target.

    Args:
        data (pl.DataFrame): DataFrame containing a 1-D time series of pre-processed and
        validated taxi rides data.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_col (str, opitonal): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.
        max_lag (int, optional): Maximum number of lag features to create. Defaults to 24.

    Returns:
        pl.DataFrame: DataFrame that contains lag features, average lag features,
        datetime features, and the corresponding target.
    """
    try:
        logger.info("Transforming the pre-processed and validated data into features and targets.")
        transformed_dfs: list[pl.DataFrame] = []
        for location_id in tqdm(sorted(data["location_id"].unique()), unit="Location ID"):
            # create the lag features
            dfs: list[pl.DataFrame] = [
                (
                    data
                    .filter(pl.col("location_id").eq(location_id))
                    .select(target_col)
                    .shift(n=lag)
                    .rename({target_col: f"lag_{lag}"})
                )
                for lag in reversed(range(1, max_lag + 1))
            ]
            df_lags: pl.DataFrame = pl.concat(dfs, how="horizontal").drop_nulls()

            # create the average lag features, i.e., window features
            start = step = 4
            dfs = [
                (
                    df_lags[:, -lag:]
                    .mean_horizontal()
                    .to_frame(name=f"avg_{lag}_lags")
                )
                for lag in reversed(range(start, max_lag + 1, step))
            ]
            df_avg_lags: pl.DataFrame = pl.concat(dfs, how="horizontal")

            # create the datetime features
            # NOTE: this pl.DataFrame also includes the 'location_id' and 'temporal_col' columns
            df_datetime: pl.DataFrame = (
                data
                .filter(pl.col("location_id").eq(location_id))
                .select(["location_id", temporal_col])
                .shift(n=-max_lag)
                .drop_nulls()
                .with_columns(
                    (
                        pl.col(temporal_col)
                        .dt.convert_time_zone(time_zone="UTC")
                        .dt.convert_time_zone(time_zone="EST")
                        .dt.hour()
                        .cast(pl.Int32)
                        .alias("hour")
                    )
                )
                .with_columns(
                    pl.when(pl.col("hour").ge(5) & pl.col("hour").lt(12)).then(1)  # morning
                    .when(pl.col("hour").ge(12) & pl.col("hour").lt(17)).then(2)  # afternoon
                    .when(pl.col("hour").ge(17) & pl.col("hour").lt(21)).then(3)  # evening
                    .otherwise(4)  # night
                    .alias("time_of_day")
                )
                .select(["location_id", temporal_col, "hour", "time_of_day"])
            )

            # horizontally concatenate the features and add the corresponding target
            transformed_data: pl.DataFrame = (
                pl.concat((df_datetime, df_avg_lags, df_lags), how="horizontal")
                .join(data, how="left", on=["location_id", temporal_col])
            )
            transformed_dfs.append(transformed_data)
        return pl.concat(transformed_dfs, how="vertical")
    except Exception as e:
        raise e


@logger.catch
def split_data(
    data: pl.DataFrame,
    test_size: int,
    temporal_col: str = data_config.temporal_column
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits data into train and test sets.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the corresponding target.
        test_size (int): Number of records for each location ID's test set.
        temporal_col (str, optional): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Train and test sets.
    """
    try:
        logger.info("Splitting the data into train and test sets.")
        # a dictionary that maps each location ID to its temporal split
        splits: dict[int, datetime] = {
            location_id: (
                data
                .filter(pl.col("location_id").eq(location_id))
                .select(temporal_col)
                .to_series()
                .max()
                - timedelta(hours=test_size)
            )
            for location_id in sorted(data["location_id"].unique())
        }

        # two empty lists, to store the train and test sets for each location ID
        train_dfs, test_dfs = [], []
        for location_id, split in tqdm(splits.items(), unit="Location ID"):
            train_dfs.append(
                data
                .filter(
                    pl.col("location_id").eq(location_id)
                    & pl.col(temporal_col).le(split)
                )
            )
            test_dfs.append(
                data
                .filter(
                    pl.col("location_id").eq(location_id)
                    & pl.col(temporal_col).gt(split)
                )
            )
        return pl.concat(train_dfs, how="vertical"), pl.concat(test_dfs, how="vertical")
    except Exception as e:
        raise e


def select_features(
    data: pl.DataFrame,
    threshold: float,
    target_col: str = data_config.target_column
) -> list[str]:
    """Selects the most relevant features based on their mutual information with the target. 

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the target.
        threshold (float): Number between 0 and 1, inclusive, that's used as a filter to
        select the most relevant features.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.

    Returns:
        list[str]: Relevant features.
    """
    try:
        logger.info("Selecting the most informative features...")
        features: list[str] = data.drop(data_config.columns).columns
        scores: np.ndarray = mutual_info_regression(data.select(features), data[target_col])
        return (
            pl.DataFrame({
                "feature": features,
                "mutual_info": (scores - scores.min()) / (scores.max() - scores.min())
            })
            .filter(
                pl.col("mutual_info").ge(pl.col("mutual_info").quantile(threshold))
            )
            .sort(by="mutual_info", descending=True)
            .select("feature")
            .to_series()
            .to_list()
        )
    except Exception as e:
        raise e


# --------------------------------------------------------------------------------------------------
# Model building utility functions
# --------------------------------------------------------------------------------------------------
def compute_metrics(y: np.ndarray | pl.Series, yhat: np.ndarray | pl.Series) -> dict[str, float]:
    """Computes the root mean squared error, RMSE, and coefficient of
    determination, R², between y and yhat.

    Args:
        y (np.ndarray | pl.Series): Observations.
        yhat (np.ndarray | pl.Series): Predictions.

    Returns:
        dict[str, float]: RMSE and R²
    """
    try:
        baseline_errors: np.ndarray | pl.Series = y - y.mean()
        sst: float = baseline_errors.dot(baseline_errors)
        model_errors: np.ndarray | pl.Series = y - yhat
        sse: float = model_errors.dot(model_errors)
        rmse: float = np.sqrt(sse / model_errors.shape[0])
        rsquared: float = 1 - (sse / sst)
        return {"rmse": round(rmse, 4), "r2": round(rsquared, 4)}
    except Exception as e:
        raise e


def get_time_series_splits(
    data: pl.DataFrame,
    temporal_col: str = data_config.temporal_column,
    n_folds: int = model_config.n_folds,
    train_pct: float = model_config.train_pct
) -> tuple[list[datetime], list[datetime]]:
    """Returns two lists, one containing the train set splits, and the other containing the
    corresponding validation set splits.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the target.
        temporal_col (str, opitonal): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.
        n_folds (int, optional): Number of train/validation splits to generate.
        Defaults to model_config.n_folds.
        train_pct (float, optional): Percentage of training data per fold.
        Defaults to model_config.train_pct.

    Returns:
        tuple[list[datetime], list[datetime]]: Train set and validation set splits.
    """
    try:
        datetimes: list[datetime] = sorted(data[temporal_col].unique())
        fold_size: int = int(round(len(datetimes) / n_folds))
        indices: list[int] = [min(fold_size * i, len(datetimes) - 1) for i in range(1, n_folds + 1)]
        train_splits: list[datetime] = [datetimes[int(train_pct * idx)] for idx in indices]
        val_splits: list[datetime] = [datetimes[idx] for idx in indices]
        return train_splits, val_splits
    except Exception as e:
        raise e


def plot_time_series_splits(
    data: pl.DataFrame,
    location_id: int,
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column
) -> None:
    """Plots the time series splits, i.e., k-fold walk-forward validation, for the input
    location ID's hourly taxi rides.

    Args:
        data (pl.DataFrame): DataFrame containing a 1-D time series of hourly taxi rides.
        location_id (str): Location ID that the time series splits will be plotted for.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_column (str, opitonal): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.
    """
    try:
        data = data.filter(pl.col("location_id").eq(location_id))
        train_splits, val_splits = data.pipe(get_time_series_splits)
        k: int = len(train_splits)

        # plot the location ID's k-fold walk-forward validation
        fig, ax = plt.subplots(k, figsize=(20, 14), sharex=True)
        fig.suptitle(f"NYC Hourly Taxi Demand, Location ID: {location_id}", fontsize=16)
        for fold, (train_split, val_split) in enumerate(zip(train_splits, val_splits)):
            y_label: str = "Number of taxi rides"
            if fold == 0:
                (
                    data
                    .filter(pl.col(temporal_col).le(train_split))
                    .select([temporal_col, target_col])
                    .to_pandas()
                    .set_index(temporal_col)
                    [target_col]
                    .plot(ax=ax[fold], style="-", label="Train Set", color="black")
                )
                (
                    data
                    .filter(
                        pl.col(temporal_col).gt(train_split)
                        & pl.col(temporal_col).le(val_split)
                    )
                    .select([temporal_col, target_col])
                    .to_pandas()
                    .set_index(temporal_col)
                    [target_col]
                    .plot(ax=ax[fold], style="--", label="Validation Set", color="black")
                )
                ax[fold].axvline(train_split, color="red", lw=3, ls="--")
                ax[fold].set_title(f"Fold {fold+1}")
                ax[fold].set_ylabel(y_label)
                ax[fold].grid(which="both", alpha=0.3)
                ax[fold].legend(loc="best", frameon=True)
            else:
                (
                    data
                    .filter(pl.col(temporal_col).le(train_split))
                    .select([temporal_col, target_col])
                    .to_pandas()
                    .set_index(temporal_col)
                    [target_col]
                    .plot(ax=ax[fold], style="-", label="Train Set", color="black")
                )
                (
                    data
                    .filter(
                        pl.col(temporal_col).gt(train_split)
                        & pl.col(temporal_col).le(val_split)
                    )
                    .select([temporal_col, target_col])
                    .to_pandas()
                    .set_index(temporal_col)
                    [target_col]
                    .plot(ax=ax[fold], style="--", label="Validation Set", color="black")
                )
                ax[fold].axvline(train_split, color="red", lw=3, ls="--")
                ax[fold].set_title(f"Fold {fold+1}")
                ax[fold].set_xlabel("Pickup-time (UTC)")
                ax[fold].set_ylabel(y_label)
                ax[fold].grid(which="both", alpha=0.3)
        plt.tight_layout()
    except Exception as e:
        raise e


def train_and_validate_model(
    data: pl.DataFrame,
    model: XGBRegressor,
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column
) -> tuple[XGBRegressor, float]:
    """Trains an ML model and returns it along with its average validation RMSE.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the corresponding target.
        model (XGBRegressor): Pre-trained ML model.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_col (str, optional): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.

    Returns:
        tuple[XGBRegressor, float]: Trained ML model and its average validation RMSE.
    """
    try:
        # a list containing the names of the features
        features: list[str] = data.drop(data_config.columns).columns

        # get the train and validation splits
        train_splits, val_splits = data.pipe(get_time_series_splits)

        # an empty list to store the model's validation metrics, one per split
        val_metrics: list[float] = []
        for train_split, val_split in zip(train_splits, val_splits):
            x_train: pl.DataFrame = (
                data
                .filter(pl.col(temporal_col).le(train_split))
                .select(features)
            )
            y_train: pl.Series = data.filter(pl.col(temporal_col).le(train_split))[target_col]
            x_val: pl.DataFrame = (
                data
                .filter(
                    pl.col(temporal_col).gt(train_split)
                    & pl.col(temporal_col).le(val_split)
                )
                .select(features)
            )
            y_val: pl.Series = (
                data
                .filter(
                    pl.col(temporal_col).gt(train_split)
                    & pl.col(temporal_col).le(val_split)
                )
                [target_col]
            )
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")
            val_metrics.append(metric)
        return model, np.mean(val_metrics).item()
    except Exception as e:
        raise e


@logger.catch
def build_model(data: pl.DataFrame) -> XGBRegressor:
    """Trains, validates, and returns an object of type, XGBRegressor.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the target.

    Returns:
        XGBRegressor: Trained and validated model.
    """
    try:
        # instantiate an object of type, XGBRegressor
        model: XGBRegressor = XGBRegressor(**model_config.xgbregressor)

        # train and validate the model
        logger.info(f"Training initiated for the '{model.__class__.__name__}'.")
        model, metric = train_and_validate_model(data, model)
        logger.info(
            f"Training complete. The {model.__class__.__name__} produced an average validation \
RMSE of {metric}."
        )
        return model
    except Exception as e:
        raise e


def hyperopt_objective(
    param_space: dict[str, Apply],
    data: pl.DataFrame,
    model: XGBRegressor
) -> dict[str, float | str]:
    """Updates the model's hyperparameters with those specified in param_space, then
    computes and returns the corresponding validation metric.

    Args:
        param_space (dict[str, Apply]): The model's hyperparameter search space.
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the target.
        model (XGBRegressor): Trained model with default hyperparameters.

    Returns:
        dict[str, float | str]: Dictionary that contains the validation metric, which is
        the objective (loss) to be optimized.
    """
    try:
        model = model.set_params(**param_space)
        metric: float = train_and_validate_model(data, model)[1]
        return {"loss": metric, "status": STATUS_OK}
    except Exception as e:
        raise e


@logger.catch
def tune_model(
    data: pl.DataFrame,
    model: XGBRegressor,
    objective_function: hyperopt_objective.__class__ = hyperopt_objective
) -> tuple[XGBRegressor, float]:
    """Returns a trained model with Bayesian-tuned hyperparameters and its corresponding
    average validation RMSE.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the target.
        model (XGBRegressor): Trained model with default hyperparameters.
        objective_function (hyperopt_objective.__class__, optional): User-defined objective
        function. Defaults to hyperopt_objective.

    Returns:
        tuple[XGBRegressor, float]: Trained model with Bayesian-tuned hyperparameters and its
        average validation RMSE.
    """
    try:
        logger.info(f"Hyperparameter tuning initiated for the {model.__class__.__name__}.")
        param_space: dict[str, Apply] = {
            "n_estimators": hp.randint("n_estimators", 100, 500),
            "max_depth": hp.randint("max_depth", 3, 10),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "min_child_weight": hp.randint("min_child_weight", 0, 10),
            "reg_alpha": hp.uniform("reg_alpha", 0, 10)
        }
        tuned_params: dict[str, float] = fmin(
            fn=partial(objective_function, data=data, model=model),
            space=param_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=Trials(),
            verbose=1
        )
        model, metric = train_and_validate_model(data, XGBRegressor(**tuned_params))
        logger.info("Hyperparameter tuning complete.")
        return model, metric
    except Exception as e:
        raise e


def save_model(model: XGBRegressor) -> None:
    """Saves model to Paths.MODEL.

    Args:
        model (XGBRegressor): ML model.
    """
    try:
        with open(Paths.MODEL, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise e


def save_model_metadata(model: XGBRegressor, metric: float) -> None:
    """Writes the model's metadata, that is, its tuned hyperparameters and corresponding
    validation metric, to a YAML file.

    Args:
        model (XGBRegressor): Trained model with Bayesian-tuned hyperparameters.
        metric: (float): The model's average validation RMSE.
    """
    try:
        # a dictionary that contains the model's metadata
        metadata: dict[str, Any] = {
            "params": {
                param: (value.item() if isinstance(value, (np.floating, np.integer)) else value)
                for param, value in model.get_params().items()
                if param in ["objective"] + model_config.hyperparams
            },
            "rmse": metric
        }

        # write the model's metadata to a YAML file.
        metadata_dir: PosixPath = Paths.MODELS_METADATA_DIR
        metadata_dir.mkdir(parents=True, exist_ok=True)
        filename: str = f"{datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S')}.yaml"
        with open(metadata_dir / filename, "w") as file:
            OmegaConf.save(metadata, file)
    except Exception as e:
        raise e


def load_model() -> XGBRegressor:
    """Loads Paths.MODEL, if it exists, otherwise the model building process is initiated,
    which trains, validates, and tunes an XGBRegressor on the latest data, saves it to
    Paths.MODEL, saves its metadata as a YAML file to Paths.MODELS_METADATA_DIR, and returns
    it as a Python object.

    Returns:
        XGBRegressor: Trained model with Bayesian-tuned hyperparameters.
    """
    try:
        # load Paths.MODEL, if it exits, otherwise start the model building process
        if Paths.MODEL.exists():
            with open(Paths.MODEL, "rb") as file:
                model: XGBRegressor = pickle.load(file)
        else:
            logger.info(
                f"~/{Paths.MODEL.parent.name}/{Paths.MODEL.name} not found. Starting the model \
building process."
            )
            data: pl.DataFrame = pl.read_parquet(Paths.DATA).pipe(transform_data)
            model, metric = tune_model(data, build_model(data))
            save_model(model)
            save_model_metadata(model, metric)
        return model
    except Exception as e:
        raise e


# --------------------------------------------------------------------------------------------------
# Inference utility functions
# --------------------------------------------------------------------------------------------------
@logger.catch
def generate_one_step_forecast(
    data: pl.DataFrame,
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column
) -> pl.DataFrame:
    """Returns a pl.DataFrame that contains each location IDs's one-step forecast, that is,
    the predicted taxi demand one hour into the future.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the corresponding target.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_col (str, optional): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.

    Returns:
        pl.DataFrame: DataFrame that contains each location ID's one-step (one-hour) forecast.
    """
    try:
        # load the model
        model: XGBRegressor = load_model()

        # a list containing the names of the features
        features: list[str] = data.drop(data_config.columns).columns

        # a dictionary that maps each location ID to its latest datetime
        latest_datetimes: dict[int, datetime] = {
            location_id: data.filter(pl.col("location_id").eq(location_id))[temporal_col].max()
            for location_id in sorted(data["location_id"].unique())
        }

        # an empty list to store each location ID's one-step forecast
        dfs: list[pl.DataFrame] = []
        for location_id, dt in tqdm(latest_datetimes.items(), unit="Location ID"):
            x: pl.DataFrame = data.filter(
                pl.col("location_id").eq(location_id)
                & pl.col(temporal_col).eq(dt)
            )

            # datetime features
            pickup_time, hour, time_of_day = (
                x
                .with_columns(
                    pl.col(temporal_col) + timedelta(hours=1)
                )
                .with_columns(
                    pl.col(temporal_col)
                    .dt.convert_time_zone(time_zone="UTC")
                    .dt.convert_time_zone(time_zone="EST")
                    .dt.hour()
                    .cast(pl.Int32)
                    .alias("hour")
                )
                .with_columns(
                    pl.when(pl.col("hour").ge(5) & pl.col("hour").lt(12)).then(1)
                    .when(pl.col("hour").ge(12) & pl.col("hour").lt(17)).then(2)
                    .when(pl.col("hour").ge(17) & pl.col("hour").lt(21)).then(3)
                    .otherwise(4)
                    .alias("time_of_day")
                )
                .select([temporal_col, "hour", "time_of_day"])
                .to_dicts()[0]
                .values()
            )

            # lag features
            max_lag: int = max(int(col.split("_")[-1]) for col in features if col.startswith("lag"))
            lags: list[int] = (
                x
                .select([col for col in features if col.startswith("lag")] + [target_col])
                .drop(f"lag_{max_lag}")
                .transpose()
                .to_series()
                .to_list()
            )

            # average lag features
            start = step = max_lag // len([col for col in x.columns if col.startswith("avg")])
            avg_lags: list[float] = [
                np.mean(lags[-lag:]) for lag in reversed(range(start, max_lag + 1, step))
            ]

            # horizontally concatenate the datetime features, average lag features, and lag features
            x = (
                pl.Series([hour, time_of_day] + avg_lags + lags, dtype=pl.Float32) # (D,)
                .to_frame() # (D, 1)
                .transpose(column_names=features) # (1, D)
                .cast(dict(zip(
                    [col for col in features if not col.startswith("avg")],
                    [pl.Int32] * len([col for col in features if not col.startswith("avg")])
                )))
            )

            # add the location ID, pickup time, and one-step forecast
            x = (
                x
                .with_columns(
                    location_id=location_id,
                    temporal_col=pickup_time,
                    forecast=max(0, int(round(model.predict(x)[0])))
                )
                .rename({"temporal_col": temporal_col})
                .select(data.drop(target_col).columns + ["forecast"])
            )

            # append the one-step forecast to the 'dfs' list
            dfs.append(x)
        return pl.concat(dfs, how="vertical")
    except Exception as e:
        raise e


def generate_multi_step_forecast(
    data: pl.DataFrame,
    forecast_horizon: int,
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column
) -> pl.DataFrame:
    """Returns a pl.DataFrame that contains a multi-step forecast for each location ID.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the corresponding target.
        forecast_horizon (int, optional): Number of time steps to forecast.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_col (str, optional): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.

    Returns:
        pd.DataFrame: DataFrame that contains each location ID's multi-step (multi-hour) forecast.
    """
    try:
        logger.info(f"Generating each location ID's {forecast_horizon}-hour forecast.")
        dfs: list[pl.DataFrame] = [
            data.pipe(generate_one_step_forecast).rename({"forecast": target_col})
        ]
        for idx in range(forecast_horizon - len(dfs)):
            dfs.append(
                dfs[idx].pipe(generate_one_step_forecast).rename({"forecast": target_col})
            )
        return (
            pl.concat(dfs, how="vertical")
            .rename({target_col: "forecast"})
            .sort(by=["location_id", temporal_col])
        )
    except Exception as e:
        raise e


def plot_record(
    data: pl.DataFrame,
    location_id: int,
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column,
    plot_forecast: bool = False
) -> Figure:
    """Plots the lag features, target, and one-step forecast for the input location ID. 

    Args:
        data (pl.DataFrame): DataFrame that contains at minimum, the temporal column,
        lag features, and the target column. The one-step forecast is optional.
        location_id (int): Input location ID.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_column (str, opitonal): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.
        plot_forecast (bool, optional): Boolean that determines if the one-step forecast
        is plotted. Defaults to False.

    Returns:
        Figure: Plotly object.
    """
    try:
        data = data.filter(pl.col("location_id").eq(location_id))
        lag_features: list[str] = [col for col in data.columns if col.startswith("lag")]
        end: datetime = data[0, temporal_col]
        lag_datetimes: list[datetime] = [
            end - timedelta(hours=lag) for lag in reversed(range(1, len(lag_features) + 1))
        ]

        # instantiate an object of type, 'Figure', with the lag features
        fig: Figure = px.line(
            x=lag_datetimes,
            y=data.select(lag_features).transpose().to_series().to_list(),
            color_discrete_sequence=["blue"],
            labels={"x": "Pick-up time (UTC)", "y": "Number of taxi rides"},
            template="plotly_dark",
            markers=True,
            title=f"Location ID: {location_id}, Pick-Up Time: {end}"
        )

        # add the target to the 'fig' object
        fig.add_scatter(
            x=[end],
            y=(
                data.select(target_col).to_series().to_list() if target_col in data.columns
                else data.select("forecast").to_series().to_list()
            ),
            line_color="green",
            mode="markers",
            marker_size=10,
            name="Target" if target_col in data.columns else "Forecast"
        )

        # add the one-step forecast to the 'fig' object
        if plot_forecast:
            fig.add_scatter(
                x=[end],
                y=data.select("forecast").to_series().to_list(),
                line_color="red",
                mode="markers",
                marker_size=10,
                name="Forecast"
            )
        return fig
    except Exception as e:
        raise e


# --------------------------------------------------------------------------------------------------
# Model evaluation utility functions
# --------------------------------------------------------------------------------------------------
def evaluate_model(
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column
) -> bool:
    """Evaluates the current model on the latest data by comparing its forecast to the naive
    forecast across different horizons. 

    Args:
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_col (str, optional): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.

    Returns:
        bool: Boolean that indicates if the current model is fine. 
    """
    try:
        # load and transform the data into features and targets
        data: pl.DataFrame = pl.read_parquet(Paths.DATA).pipe(transform_data)

        # two empty lists to store the forecasting metrics
        model_metrics: list[float] = []
        naive_metrics: list[float] = []
        for eval_size in range(1, model_config.test_size + 1):
            # split the data into a train and evaluation set
            train_data, eval_data = data.pipe(split_data, eval_size)

            # create a dictionary that maps each location ID to its last known train set value
            naive_forecast: dict[int, int] = (
                train_data
                .select(["location_id", temporal_col, target_col])
                .join(
                    other=(
                        train_data
                        .group_by("location_id", maintain_order=True)
                        .agg(pl.col(temporal_col).max())
                    ),
                    how="inner",
                    on=["location_id", temporal_col]
                )
                .to_pandas()
                .set_index("location_id")
                [target_col]
                .to_dict()
            )

            # update the evaluation set to include the model's forecast and naive forecast
            eval_data = (
                train_data
                .pipe(generate_multi_step_forecast, eval_size)
                .with_columns(
                    pl.col("location_id")
                    .map_elements(
                        lambda location_id: naive_forecast.get(location_id), return_dtype=pl.Int32
                    )
                    .alias("naive_forecast")
                )
                .select(["location_id", temporal_col, "naive_forecast", "forecast"])
                .join(
                    other=eval_data.select(["location_id", temporal_col, target_col]),
                    how="left",
                    on=["location_id", temporal_col]
                )
            )

            # compute the forecasting metrics and append them to their respective lists
            model_metrics.append(
                compute_metrics(eval_data[target_col], eval_data["forecast"]).get("r2")
            )
            naive_metrics.append(
                compute_metrics(eval_data[target_col], eval_data["naive_forecast"]).get("r2")
            )
        return pl.Series(model_metrics).mean() > max(0.8, pl.Series(naive_metrics).mean())
    except Exception as e:
        raise e
