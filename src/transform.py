"""This module provides functionality to transform pre-processed data into features and labels."""

import pandas as pd

from tqdm import tqdm

from src.feature_store_api import HOPSWORKS_CONFIG, get_feature_group
from src.logger import logger


def tabularize_data(data: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    """Converts a pd.DataFrame containing the validated and pre-processed NYC
    taxi demand data to a pd.DataFrame containing datetime features, window
    features (average lag features), lag features, and the corresponding target

    Args:
        data (pd.DataFrame): Validated and pre-processed data
        max_lag (int, optional): Maximum of lag features to create. Defaults to 24.

    Returns:
        pd.DataFrame: Machine learning-ready data
    """
    try:
        logger.info("Transforming the NYC taxi demand data into features and labels.")
        # an empty list to store the tabular, ML-ready pd.DataFrames, one per location ID
        tabular_dfs: list[pd.DataFrame] = []
        for location_id in tqdm(sorted(data["location_id"].unique())):
            # create the lag features
            lag_dfs: list[pd.DataFrame] = [
                (
                    data
                    .set_index("pickup_time")
                    .query(f"location_id == {location_id}")
                    .drop("location_id", axis=1)
                    .shift(periods=lag)
                    .rename({"n_rides": f"lag_{lag}"}, axis=1)
                )
                for lag in reversed(range(1, max_lag + 1))
            ]
            df_lags: pd.DataFrame = pd.concat(lag_dfs, axis=1).dropna()

            # create the window features, that is, average lag features
            start = step = 4
            window_dfs: list[pd.DataFrame] = [
                df_lags.iloc[:, -lag:].mean(axis=1).to_frame(name=f"avg_{lag}_lags")
                for lag in reversed(range(start, max_lag + 1, step))
            ]
            df_windows: pd.DataFrame = pd.concat(window_dfs, axis=1)

            # a list of output column names
            output_cols: list[str] = (
                ["location_id", "pickup_time", "day_of_week", "hour"] +
                df_windows.columns.tolist() +
                df_lags.columns.tolist() +
                ["target"]
            )

            # horizontally concatenate the window and lag features
            df_tabular: pd.DataFrame = pd.concat((df_windows, df_lags), axis=1)

            # create two temporal features, 'day_of_week' and 'hour', and ...
            # add the location ID and corresponding target to the 'df_tabular' pd.DataFrame
            df_tabular = (
                df_tabular
                .assign(
                    day_of_week=df_tabular.index.day_of_week,
                    hour=df_tabular.index.hour,
                    location_id=location_id,
                    target=(
                        data
                        .set_index("pickup_time")
                        .query(f"location_id == {location_id}")
                        .loc[df_tabular.index, "n_rides"]
                    )
                )
                .reset_index()
                [output_cols]
            )
            tabular_dfs.append(df_tabular)
        return pd.concat(tabular_dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e


def fetch_and_transform() -> pd.DataFrame:
    """Fetches the latest NYC taxi demand data from Hopsworks and transforms it into a
    machine learning-ready dataset of features and labels

    Returns:
        pd.DataFrame: Machine learning-ready data
    """
    try:
        # load the latest NYC taxi demand data from Hopsworks
        data: pd.DataFrame = get_feature_group().read()
        if data.empty:
            logger.info(
                f"Data not fetched. No data exists for Project Name: '{HOPSWORKS_CONFIG.project}', \
Feature Group: '{HOPSWORKS_CONFIG.feature_group.name}'"
            )
        else:
            # transform the latest NYC taxi demand data into ML-ready features and labels
            return (
                data
                .assign(unix_time_ms=pd.to_datetime(data["unix_time_ms"], unit="ms"))
                .rename({"unix_time_ms": "pickup_time"}, axis=1)
                .sort_values(by=["location_id", "pickup_time"])
                .reset_index(drop=True)
                .pipe(tabularize_data)
            )
    except Exception as e:
        raise e
