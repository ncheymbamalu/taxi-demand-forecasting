"""This module provides functionality for uploading the latest pre-processed data to Hopsworks."""

from datetime import datetime, timezone

import pandas as pd

from src.feature_store_api import HOPSWORKS_CONFIG, get_feature_group
from src.ingest import download_data
from src.logger import logger


def upload_data() -> None:
    """Downloads raw data from the 'NYC trip data' URL, validates, pre-processes,
    and writes it to Hopsworks
    """
    try:
        # get the raw data's recorded year and month(s)
        end: pd.Timestamp = pd.Timestamp(datetime.now(timezone.utc)).replace(tzinfo=None).floor("H")
        start: pd.Timestamp = end - pd.Timedelta(days=14)
        year: int = (end - pd.Timedelta(days=366)).year
        months: list[int] = [
            (pd.Timestamp(f"{year}-{end.month}-01") - pd.Timedelta(days=1)).month,
            end.month
        ]

        # download, validate, and pre-process the raw data
        dfs: list[pd.DataFrame] = [download_data(year, month) for month in months]
        data: pd.DataFrame = pd.concat(dfs, axis=0, ignore_index=True)
        data = (
            data
            .assign(
                pickup_time=data["pickup_time"] + pd.Timedelta(days=366),
                unix_time_ms=(data["pickup_time"] + pd.Timedelta(days=366)).astype(int) // 1_000_000
            )
            .query(f"pickup_time >= '{start}' & pickup_time <= '{end}'")
            .sort_values(by=["location_id", "pickup_time"])
            .drop_duplicates(subset=["location_id", "pickup_time"], keep="first")
            .reset_index(drop=True)
            [["location_id", "unix_time_ms", "n_rides"]]
        )

        # write the validated and pre-processed data to Hopsworks
        logger.info(
            f"Uploading the latest batch of NYC taxi demand data to Hopsworks, \
Project Name: '{HOPSWORKS_CONFIG.project}', Feature Group: '{HOPSWORKS_CONFIG.feature_group.name}'"
        )
        get_feature_group().insert(data, write_options={"wait_for_job": False})
    except Exception as e:
        raise e


if __name__ == "__main__":
    upload_data()
