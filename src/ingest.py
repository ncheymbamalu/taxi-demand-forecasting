"""A script that downloads, validates, and pre-processes raw NYC taxi rides data"""

import os

from datetime import datetime, timezone

import pandas as pd
import requests

from requests import Response
from tqdm import tqdm

from src.logger import logging
from src.paths import PathConfig


def validate_data(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Returns a null value and duplicate-free pd.DataFrame of irregularly sampled
    NYC taxi rides between 'start' and 'end' inclusive

    Args:
        data (pd.DataFrame): Raw data
        start (pd.Timestamp): Starting timestamp that's used to filter the raw data
        end (pd.Timestamp): Ending timestamp that's used to filter the raw data

    Returns:
        pd.DataFrame: Validated data
    """
    try:
        return (
            data
            .rename({"PULocationID": "location_id"}, axis=1)
            .assign(
                pickup_time=pd.to_datetime(data["tpep_pickup_datetime"]) + pd.Timedelta(days=366)
            )
            .query(f"pickup_time >= '{start}' and pickup_time <= '{end}'")
            .dropna(subset=["location_id", "pickup_time"])
            .drop_duplicates(subset=["location_id", "pickup_time"], keep="first")
            .sort_values(["location_id", "pickup_time"])
            .reset_index(drop=True)
            [["location_id", "pickup_time"]]
        )
    except Exception as e:
        raise e


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Returns a pd.DataFrame containing NYC taxi rides recorded at an hourly frequency,
    for each location ID

    Args:
        data (pd.DataFrame): Null value and duplicate-free data containing irregularly
        sampled NYC taxi rides

    Returns:
        pd.DataFrame: Pre-processed data
    """
    try:
        # a list of pre-processed pd.DataFrames, one per location ID
        dfs: list[pd.DataFrame] = [
            (
                data
                .query(f"location_id == {location_id}")
                .assign(pickup_time=pd.to_datetime(data["pickup_time"]).dt.floor("H"))
                .groupby("pickup_time", as_index=False)
                .agg(n_rides=("pickup_time", "size"))
                .set_index("pickup_time")
                .asfreq("H")
                .fillna(0)
                .astype({"n_rides": int})
                .assign(location_id=location_id)
                .reset_index()
                [["location_id", "pickup_time", "n_rides"]]
            )
            for location_id in tqdm(sorted(data["location_id"].unique()))
        ]
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e


def download_data() -> pd.DataFrame:
    """Downloads a single file of raw data from the 'NYC trip data' URL, validates
    and pre-processes it, and returns a pd.DataFrame containing NYC taxi rides
    recorded at an hourly frequency, for each location ID

    Returns:
        pd.DataFrame: Validated and pre-processed data
    """
    try:
        end: pd.Timestamp = pd.Timestamp(datetime.now(timezone.utc)).replace(tzinfo=None).floor("H")
        start: pd.Timestamp = end - pd.Timedelta(days=14)
        year: int = (end - pd.Timedelta(days=366)).year
        month: int = (end - pd.Timedelta(days=366)).month
        filename: str = f"yellow_tripdata_{year}-{month:02d}.parquet"
        response: Response = requests.get(os.path.join(PathConfig.RAW_DATA_URL, filename))
        if response.status_code == 200:
            logging.info(
                "Downloading, validating, and pre-processing %s.",
                os.path.join(PathConfig.RAW_DATA_URL, filename)
            )
            return (
                pd.read_parquet(os.path.join(PathConfig.RAW_DATA_URL, filename))
                .pipe(validate_data, start, end)
                .pipe(preprocess_data)
            )
        else:
            logging.info(
                "%s is not available or download.", os.path.join(PathConfig.RAW_DATA_URL, filename)
            )
    except Exception as e:
        raise e
