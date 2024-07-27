"""A script that transforms hourly NYC taxi demand data into a tabular, ML-ready dataset"""

import pandas as pd
import polars as pl

from hsfs.feature_group import FeatureGroup
from tqdm import tqdm

from src.feature_store_api import get_feature_group
from src.logger import logging


def tabularize_data(data: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    """Converts a pd.DataFrame containing a 1-D time series of validated and 
    pre-processed NYC taxi rides recorded at an hourly frequency, to a pd.DataFrame 
    containing datetime features, window features (average lag features), lag 
    features, and the corresponding target

    Args:
        data (pd.DataFrame): Validated and pre-processed data
        max_lag (int, optional): Maximum number of lag features to create. Defaults to 24.

    Returns:
        pd.DataFrame: ML-ready dataset containing datetime features, window features 
        (average lag features), lag features, and the corresponding target
    """
    try:
        logging.info("Transforming the hourly taxi demand data into features and labels.")
        # an empty list to store the tabular pd.DataFrames, one per location ID
        dfs: list[pd.DataFrame] = []
        for location_id in tqdm(data["location_id"].unique()):
            # create the lag features
            lags: list[pd.DataFrame] = [
                (
                    data.query(f"location_id == {location_id}")
                    .drop("location_id", axis=1)
                    .set_index("pickup_datetime")
                    .shift(periods=lag)
                    .rename({"rides": f"lag_{lag}"}, axis=1)
                )
                for lag in reversed(range(1, max_lag + 1))
            ]

            # create the window features, i.e., average lag features
            avg_lags: list[pd.DataFrame] = [
                (
                    pd.concat(lags, axis=1)
                    .dropna()
                    .iloc[:, -lag:]
                    .mean(axis=1)
                    .to_frame()
                    .rename({0: f"avg_{lag}_lags"}, axis=1)
                )
                for lag in reversed(range(4, max_lag + 1, 4))
            ]

            # horizontally concatenate the window features and lag features
            tabular_data: pd.DataFrame = pd.concat(avg_lags + lags, axis=1).dropna()

            # a list of all the feature names
            features: list[str] = (
                ["hour", "day_of_week"]
                + [col for col in tabular_data.columns if col.startswith("avg")]
                + [col for col in tabular_data.columns if col.startswith("lag")]
            )

            # final processing
            # (1) add two datetime features, based on the 'pickup_datetime' index, ...
            # one that extracts the hour and one that extracts the day of the week
            # (2) add the location ID
            # (3) add the corresponding target
            # (4) reset the index so that the 'pickup_datetime' index becomes a column
            # (5) re-arrange the resulting columns
            tabular_data = tabular_data.assign(
                hour=tabular_data.index.hour,
                day_of_week=tabular_data.index.day_of_week,
                location_id=location_id,
                target=(
                    data.query(f"location_id == {location_id}")
                    .set_index("pickup_datetime")
                    .loc[tabular_data.index, "rides"]
                ),
            ).reset_index()[["location_id", "pickup_datetime"] + features + ["target"]]
            dfs.append(tabular_data)
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e
    

def fetch_and_transform() -> pd.DataFrame:
    """Fetches the latest validated and pre-processed NYC taxi demand data from the 
    Hopsworks 'taxi_demand_forecasting' project's 'univariate_time_series' Feature Group, 
    transforms it into an ML-ready dataset containing datetime features, window features 
    (average lag features), lag features, and the target, and returns it as a pd.DataFrame 
    
    Returns:
        pd.DataFrame: ML-ready dataset containing datetime features, window features 
        (average lag features), lag features, and the corresponding target
    """
    try:
        # connect to the project's Feature Group
        feature_group: FeatureGroup = get_feature_group()
        
        # load the latest NYC taxi demand data from the project's Feature Group
        data: pd.DataFrame = feature_group.read()
        if data.empty:
            logging.info(
                "No data exists for the '%s' project's '%s' feature group. Exiting the function.",
                feature_group._get_project_name(),
                feature_group.name,
            )
        else:
            # transform the latest NYC taxi demand data into ML-ready features and labels
            return (
                pl.from_pandas(data)
                .with_columns(pl.from_epoch(pl.col("unix_epoch_ms"), time_unit="ms"))
                .rename({"unix_epoch_ms": "pickup_datetime"})
                .sort(["location_id", "pickup_datetime"])
                .to_pandas()
                .pipe(tabularize_data)
            )
    except Exception as e:
        raise e
