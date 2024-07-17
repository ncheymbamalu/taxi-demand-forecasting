"""A script that downloads, validates, and pre-processes raw NYC taxi rides data"""

import calendar
import os

import pandas as pd
import requests

from omegaconf import OmegaConf
from requests import Response
from tqdm import tqdm

from src.config import Config, load_config
from src.logger import logging

INGEST_CONFIG: dict[str, list[str]] = OmegaConf.to_container(load_config().ingest)


def validate_data(data: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Returns a pd.DataFrame containing only records (taxi rides) for the
    specified year and month, that's also free of null values and duplicates

    Args:
        data (pd.DataFrame): Raw data
        year (int): Raw data's recorded year
        month (int): Raw data's recorded month

    Returns:
        pd.DataFrame: Validated data
    """
    try:
        start: pd.Timestamp = pd.Timestamp(f"{year}-{month:02d}-01")
        last_day: int = calendar.monthrange(year, month)[1]
        end: pd.Timestamp = pd.Timestamp(f"{year}-{month:02d}-{last_day}") + pd.Timedelta(days=1)
        return (
            data.assign(pickup_datetime=pd.to_datetime(data["tpep_pickup_datetime"]))[
                (pd.to_datetime(data["tpep_pickup_datetime"]) >= start)
                & (pd.to_datetime(data["tpep_pickup_datetime"]) < end)
            ]
            .dropna(subset=["tpep_pickup_datetime", "PULocationID"])
            .drop_duplicates(subset=["tpep_pickup_datetime", "PULocationID"], keep="first")
            .rename({"PULocationID": "location_id"}, axis=1)
            .sort_values(["location_id", "pickup_datetime"])
            .reset_index(drop=True)[INGEST_CONFIG.get("raw_columns")]
        )
    except Exception as e:
        raise e


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Returns a pd.DataFrame containing taxi rides recorded at
    regularly spaced hourly timestamps, for each unique location ID

    Args:
        data (pd.DataFrame): Validated data

    Returns:
        pd.DataFrame: Pre-rocessed data
    """
    try:
        # a list of pre-processed pd.DataFrames, one per location ID
        dfs: list[pd.DataFrame] = [
            (
                data.query(f"location_id == {location_id}")
                .assign(pickup_datetime=data["pickup_datetime"].dt.floor("H"))
                .groupby("pickup_datetime", as_index=False)
                .agg(rides=("pickup_datetime", "size"))
                .set_index("pickup_datetime")
                .asfreq("H")
                .fillna(0)
                .assign(location_id=location_id)
                .reset_index()[INGEST_CONFIG.get("processed_columns")]
            )
            for location_id in tqdm(data["location_id"].unique())
        ]
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e


def download_file(year: int, month: int) -> pd.DataFrame:
    """Downloads a single file of raw data from the 'NYC trip data' URL,
    validates, pre-processes, and returns it as a pd.DataFrame containing
    taxi rides recorded at regularly spaced hourly timestamps, for each
    unique location ID

    Args:
        year (int): Raw data's recorded year
        month (int): Raw data's recorded month
    """
    try:
        filename: str = f"yellow_tripdata_{year}-{month:02d}.parquet"
        response: Response = requests.get(os.path.join(Config.URL, filename))
        if response.status_code == 200:
            logging.info(
                "Downloading, validating, and pre-processing %s", os.path.join(Config.URL, filename)
            )
            return (
                pd.read_parquet(os.path.join(Config.URL, filename))
                .pipe(validate_data, year, month)
                .pipe(preprocess_data)
            )
        else:
            logging.info("%s is not available", os.path.join(Config.URL, filename))
    except Exception as e:
        raise e
