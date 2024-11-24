"""This module provides functionality to download, validate, and pre-process raw data."""

import calendar
import os

import pandas as pd
import requests

from requests import Response
from tqdm import tqdm

from src.config import Paths
from src.logger import logger


def validate_data(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Returns a pd.DataFrame of irregularly sampled NYC taxi rides between 'start'
    and 'end' inclusive, that's free of null values and duplicate records

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
            .query(f"tpep_pickup_datetime >= '{start}' and tpep_pickup_datetime < '{end}'")
            .dropna(subset=["PULocationID", "tpep_pickup_datetime"])
            .drop_duplicates(subset=["PULocationID", "tpep_pickup_datetime"], keep="first")
            .sort_values(["PULocationID", "tpep_pickup_datetime"])
            .rename({"PULocationID": "location_id", "tpep_pickup_datetime": "pickup_time"}, axis=1)
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


def download_data(year: int, month: int) -> pd.DataFrame:
    """Downloads raw data from the 'NYC trip data' URL, then validates, pre-processes,
    and returns it as a pd.DataFrame containing NYC taxi rides recorded at an hourly
    frequency, for each location ID

    Args:
        year (int): Raw data's recorded year
        month (int): Raw data's recorded month

    Returns:
        pd.DataFrame: Validated and pre-processed data if the raw data's URL is valid,
        otherwise an empty pd.DataFrame
    """
    try:
        filename: str = f"yellow_tripdata_{year}-{month:02d}.parquet"
        response: Response = requests.get(os.path.join(Paths.RAW_DATA_URL, filename))
        if response.status_code == 200:
            logger.info(
                f"Downloading, validating, and pre-processing \
{os.path.join(Paths.RAW_DATA_URL, filename)}."
            )
            start: pd.Timestamp = pd.Timestamp(f"{year}-{month:02d}-01")
            last_day_of_month: int = calendar.monthrange(year, month)[1]
            end: pd.Timestamp = (
                pd.Timestamp(f"{year}-{month:02d}-{last_day_of_month}") + pd.Timedelta(days=1)
            )
            return (
                pd.read_parquet(os.path.join(Paths.RAW_DATA_URL, filename))
                .pipe(validate_data, start, end)
                .pipe(preprocess_data)
            )
        logger.info(
            f"Invalid request. {os.path.join(Paths.RAW_DATA_URL, filename)} is not available to \
download."
        )
        return pd.DataFrame()
    except Exception as e:
        raise e
