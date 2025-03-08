"""This function contains utility/helper functions."""

import os
import pickle

from datetime import datetime, timedelta
from functools import partial

import httpx
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from httpx import Response
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import Apply
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
from xgboost import XGBRegressor

from src.config import Paths, data_config, model_config
from src.logger import logger


# --------------------------------------------------------------------------------------------------
# Data-related utility functions
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
        threshold (float): Percentile threshold that's used as a filter to identify the
        most relevant features.
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
                pl.col("mutual_info").gt(pl.col("mutual_info").quantile(threshold))
            )
            .sort(by="mutual_info", descending=True)
            .select("feature")
            .to_series()
            .to_list()
        )
    except Exception as e:
        raise e


# --------------------------------------------------------------------------------------------------
# Model building-related utility functions
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
        return model, np.mean(val_metrics)
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
    target_col: str = data_config.target_column,
    objective_function: hyperopt_objective.__class__ = hyperopt_objective
) -> XGBRegressor:
    """Returns a trained model with Bayesian-tuned hyperparameters.

    Args:
        data (pl.DataFrame): DataFrame that contains lag features, average lag features,
        datetime features, and the target.
        model (XGBRegressor): Trained model with default hyperparameters.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        objective_function (hyperopt_objective.__class__, optional): User-defined objective
        function. Defaults to hyperopt_objective.

    Returns:
        XGBRegressor: Trained model with Bayesian-tuned hyperparameters.
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
        model = XGBRegressor(**tuned_params)
        features: list[str] = data.drop(data_config.columns).columns
        model.fit(data.select(features), data[target_col])
        logger.info("Hyperparameter tuning complete.")
        return model
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
