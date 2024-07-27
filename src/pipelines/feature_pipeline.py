"""A script that uploads the latest NYC taxi demand data to Hopsworks"""

import os

from datetime import datetime, timezone

import hopsworks
import pandas as pd

from src.feature_store_api import HOPSWORKS_CONFIG
from src.ingest import fetch_and_preprocess
from src.logger import logging

# specify the data's starting and ending timestamps
END: pd.Timestamp = pd.Timestamp(datetime.now(timezone.utc)).floor("H") - pd.Timedelta(days=366)
START: pd.Timestamp = END - pd.Timedelta(days=28)


def upload_data() -> None:
    # fetch, validate, pre-process, and filter the raw data
    df: pd.DataFrame = fetch_and_preprocess(START, END)

    # upload the 'df' pd.DataFrame's data to Hopsworks
    logging.info(
        "Uploading the latest taxi demand data to Hopsworks, Project Name: %s, Feature Group: %s",
        HOPSWORKS_CONFIG.get("project"), HOPSWORKS_CONFIG.get("feature_group").get("name")
    )
    (  
    hopsworks
    .login(
        project=HOPSWORKS_CONFIG.get("project"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    .get_feature_store()
    .get_or_create_feature_group(**HOPSWORKS_CONFIG.get("feature_group"))
    .insert(df, write_options={"wait_for_job": False})
)


if __name__ == "__main__":
    upload_data()
