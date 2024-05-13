"""Data ingestion"""

import calendar
import glob
import os
import time

from datetime import timedelta
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
import requests

from requests.models import Response
from tqdm import tqdm

from src.logger import logging

PROJECT_DIR: PosixPath = Path(".").parent.resolve().parent
RAW_DATA_DIR: str = os.path.join(PROJECT_DIR, "data", "raw")
PREPROCESSED_DATA_DIR: str = RAW_DATA_DIR.replace("raw", "preprocessed")


def download_file(year: int, month: int) -> str:
    """Downloads raw data from a url, writes it locally to './data/raw' 
    as a parquet file, and returns the local parquet file path

    Args:
        year (int): The raw data's recorded year
        month (int): The raw data's recorded month

    Returns:
        str: Local file path where the raw data is stored
    """
    url: str = os.path.join(
        "https://d37ci6vzurychx.cloudfront.net/trip-data",
        f"yellow_tripdata_{year}-{month:02d}.parquet",
    )
    response: Response = requests.get(url)
    if response.status_code == 200:
        raw_data_path: str = f"{RAW_DATA_DIR}/rides_{year}-{month:02d}{Path(url).suffix}"
        open(raw_data_path, "wb").write(response.content)
        return raw_data_path
    logging.info("No available data for %s", f"{year}-{month:02d}")


def preprocess_and_validate(year: int, month: int) -> None:
    """Reads in the raw data's parquet file as a pd.DataFrame, pre-processes 
    it, validates it, writes it to './data/preprocessed' as a parquet file

    Args:
        year (int): The raw data's recorded year
        month (int): The raw data' recorded month

    Returns:
        pd.DataFrame: Pre-processed and validated data that contains several
        univariate time series, one per location ID
    """
    # download the raw data
    raw_data_path: str | None = download_file(year, month)
    if raw_data_path:
        logging.info("Pre-processing and validating data for %s", f"{year}-{month:02d}")
        raw_data: pd.DataFrame = pd.read_parquet(raw_data_path)

        # pre-process and validate the raw data
        start: pd.Timestamp = pd.Timestamp(f"{year}-{month:02d}-01")
        last_day_of_month: int = calendar.monthrange(year, month)[-1]
        end: pd.Timestamp = pd.Timestamp(f"{year}-{month:02d}-{last_day_of_month}")
        preprocessed_data: pd.DataFrame = (
            raw_data.loc[
                (raw_data["tpep_pickup_datetime"] >= start) &
                (raw_data["tpep_pickup_datetime"] < end + pd.Timedelta(days=1))
            ]
            .sort_values(["PULocationID", "tpep_pickup_datetime"])
            .rename(
                {
                    "tpep_pickup_datetime": "pickup_datetime", 
                    "PULocationID": "pickup_location_id"
                },
                axis=1
            )
            .reset_index(drop=True)
            [["pickup_datetime", "pickup_location_id"]]
        )

        # a list of pre-processed and validated pd.DataFrames, one per location ID
        location_id_dfs: list[pd.DataFrame] = [
            (
                preprocessed_data
                .assign(pickup_hour=preprocessed_data["pickup_datetime"].dt.floor("H"))
                .groupby(["pickup_location_id", "pickup_hour"], as_index=False).size()
                .rename({"size": "n_taxi_rides"}, axis=1)
                .query(f"pickup_location_id == {location_id}")
                # NOTE: set the index to the 'pickup_hour' column, and ...
                # re-sample to ensure that all hourly timestamp increments are included
                .set_index("pickup_hour").asfreq("H")
                .reset_index()
                .assign(pickup_location_id=location_id)
                .fillna(0)
                [["pickup_location_id", "pickup_hour", "n_taxi_rides"]]
            )
            for location_id in tqdm(sorted(preprocessed_data["pickup_location_id"].unique()))
        ]

        # vertically concatenate the list of pre-processed location ID pd.DataFrames, and ...
        # write locally to './data/preprocessed' as a parquet file
        (
            pd.concat(location_id_dfs, axis=0, ignore_index=True)
            .to_parquet(raw_data_path.replace("raw", "preprocessed"))
        )


def ingest_data(year: int, months: list[int] = np.arange(1, 13).tolist()) -> pd.DataFrame:
    """Ingests a list of parquet files for a given year, where each contains raw data that 
    will be pre-processed, validated, and written to './data/preprocessed' as a parquet.  
    Then, all parquet files containing pre-processed and validated data are read in as a 
    list of pd.DataFrames, vertically concatenated, and returned as a single pd.DataFrame. 

    Args:
        year (int): The recorded year for all the parquet files containing raw data
        months (list[int]): A list of months (integers) for which raw data will be 
        downloaded, pre-processed, and validated.  Defaults to np.arange(1, 13).tolist()

    Returns:
        pd.DataFrame: Pre-processed and validated data that contains several 
        univariate time series, one per pickup location ID
    """
    try:
        for month in tqdm(months):
            if os.path.exists(
                os.path.join(PREPROCESSED_DATA_DIR, f"rides_{year}-{month:02d}.parquet")
            ):
                logging.info("%s exists. Skipping download.", f"rides_{year}-{month:02d}.parquet")
            else:
                preprocess_and_validate(year, month)
        time.sleep(3)
        preprocessed_dfs: list[pd.DataFrame] = [
            pd.read_parquet(path)
            for path in sorted(glob.glob(os.path.join(PREPROCESSED_DATA_DIR, "*.parquet")))
        ]
        return pd.concat(preprocessed_dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e


def fetch_synthetic_batch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a synthetic batch of data

    Args:
        start (pd.Timestamp): Starting timestamp of the synthetic batch of data
        end: (pd.Timestamp): Ending timestamp of the synthetic batch of data
    
    Returns:
        pd.DataFrame: Synthetic batch of data. NOTE: the data is 'synthetic' because 
        the actual timestamps are shifted forward in time by one year, to make it seem 
        like the data is being fetched in real time.
    """
    try:
        # ingest the pre-processed and validated data
        data: pd.DataFrame = ingest_data(2022)

        # create the synthetic batch of data
        synthetic_data: pd.DataFrame = (
            data
            .assign(pickup_hour=data["pickup_hour"] + timedelta(days=7*52))
            .loc[
                (data["pickup_hour"] + timedelta(days=7*52) >= start) &
                (data["pickup_hour"] + timedelta(days=7*52) <= end)
            ]
            .reset_index(drop=True)
        )
        return synthetic_data
    except Exception as e:
        raise e
