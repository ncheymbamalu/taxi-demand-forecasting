"""A script that uploads the latest NYC taxi demand data to Hopsworks"""

from datetime import datetime, timezone

import pandas as pd
import polars as pl

from hsfs.feature_group import FeatureGroup

from src.feature_store_api import get_feature_group
from src.ingest import fetch_and_preprocess
from src.logger import logging

# specify the data's starting and ending timestamps
END: pd.Timestamp = pd.Timestamp(datetime.now(timezone.utc)).floor("H") - pd.Timedelta(days=366)
START: pd.Timestamp = END - pd.Timedelta(days=28)


def main() -> None:
    # fetch, validate, pre-process, and filter the raw data
    df: pd.DataFrame = fetch_and_preprocess(START, END)

    # upload the validated/pre-processed/filtered data to Hopsworks
    feature_group: FeatureGroup = get_feature_group()
    logging.info(
        "Uploading NYC taxi demand data to Hopsworks, Project Name: %s, Feature Group: %s",
        feature_group._get_project_name(),
        feature_group.name,
    )
    feature_group.insert(df, write_options={"wait_for_job": False})


if __name__ == "__main__":
    main()
