import os

import hopsworks
import pandas as pd

from omegaconf import DictConfig

from src.ingest import download_data
from src.logger import logging
from src.paths import load_config


HOPSWORKS_CONFIG: DictConfig = load_config().hopsworks


def upload_data() -> None:
    """Downloads raw data from the 'NYC trip data' URL, validates,
    pre-processes, and writes it to Hopsworks
    """
    try:
        # download, validate, and pre-process the raw data
        df: pd.DataFrame = download_data()
        df = (
            df
            .assign(pickup_time=df["pickup_time"].astype(int) // 1_000_000)
            .rename({"pickup_time": "unix_time_ms"}, axis=1)
            .drop_duplicates(subset=["location_id", "unix_time_ms"], keep="first")
            .sort_values(by=["location_id", "unix_time_ms"])
            .reset_index(drop=True)
        )
        
        # write the 'df' pd.DataFrame's data to Hopsworks
        logging.info(
            "Uploading the latest NYC taxi demand data to Hopsworks, Project Name: %s, \
Feature Group: %s",
            HOPSWORKS_CONFIG.project, HOPSWORKS_CONFIG.feature_group.name
        )
        (
            hopsworks
            .login(
                project=HOPSWORKS_CONFIG.project,
                api_key_value=os.getenv("HOPSWORKS_API_KEY")
            )
            .get_feature_store()
            .get_or_create_feature_group(**HOPSWORKS_CONFIG.feature_group)
            .insert(df, write_options={"wait_for_job": False})
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    upload_data()
