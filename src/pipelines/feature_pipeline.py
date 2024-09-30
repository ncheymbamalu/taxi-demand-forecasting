"""A script that uploads the latest NYC taxi demand data to Hopsworks"""

import pandas as pd

from src.feature_store_api import HOPSWORKS_CONFIG, get_feature_group
from src.ingest import download_data
from src.logger import logging


def upload_data() -> None:
    """Downloads raw data from the 'NYC trip data' URL, validates,
    pre-processes, and writes it to Hopsworks
    """
    try:
        # download, validate, and pre-process the raw data
        data: pd.DataFrame = download_data()
        data = (
            data
            .assign(pickup_time=data["pickup_time"].astype(int) // 1_000_000)
            .rename({"pickup_time": "unix_time_ms"}, axis=1)
            .drop_duplicates(subset=["location_id", "unix_time_ms"], keep="first")
            .sort_values(by=["location_id", "unix_time_ms"])
            .reset_index(drop=True)
        )

        # write the validated and pre-processed data to Hopsworks
        logging.info(
            "Uploading the latest NYC taxi demand data to Hopsworks, Project Name: %s, \
Feature Group: %s",
            HOPSWORKS_CONFIG.project, HOPSWORKS_CONFIG.feature_group.name
        )
        get_feature_group().insert(data, write_options={"wait_for_job": False})
    except Exception as e:
        raise e


if __name__ == "__main__":
    upload_data()
