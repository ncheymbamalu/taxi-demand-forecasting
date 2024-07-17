"""A scrip to upload data to Hopsworks"""

from datetime import datetime, timezone

import pandas as pd
import polars as pl

from hsfs.feature_group import FeatureGroup

from src.feature_store_api import get_feature_group
from src.logger import logging
from src.ingest import download_file


# starting and ending timestamps that will be used to filter the fetched data
END: pd.Timestamp = pd.Timestamp(datetime.now(timezone.utc)).floor("H") - pd.Timedelta(days=366)
START: pd.Timestamp = END - pd.Timedelta(days=28)


def fetch_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Returns a validated, pre-processed, and timestamp filtered pd.DataFrame

    Args:
        start (pd.Timestamp): Starting timestamp used to filter the data
        end (pd.Timestamp): Ending timestamp used to filter the data

    Returns:
        pd.DataFrame: Validated, pre-processed, and timestamp filtered data
    """
    try:
        # a list of pre-processed pd.DataFrames, one for each (year, month) pair
        dfs: list[pd.DataFrame] = [
            download_file(year, month) 
            for year, month in zip([start.year, end.year], [start.month, end.month])
        ]
        return (
            pl.concat([pl.from_pandas(df) for df in dfs], how="vertical")
            .filter(
                pl.col("pickup_datetime").dt.replace_time_zone("UTC").ge(start),
                pl.col("pickup_datetime").dt.replace_time_zone("UTC").le(end)
            )
            .with_columns(
                pl.col("pickup_datetime")
                .dt.replace_time_zone("UTC")
                .dt.offset_by("1y")
                .dt.epoch(time_unit="ms")
            )
            .rename({"pickup_datetime": "unix_epoch_ms"})
            .to_pandas()
        )
    except Exception as e:
        raise e
    

def main() -> None:
    # fetch the data
    df: pd.DataFrame = fetch_data(START, END)
    
    # upload the fetched data to Hopswork
    feature_group: FeatureGroup = get_feature_group()
    logging.info(
        "Uploading data to Hopsworks, Project Name: %s, Feature Group: %s", 
        feature_group._get_project_name(), feature_group.name
    )
    feature_group.insert(df, write_options={"wait_for_job": False})
    

if __name__ == "__main__":
    # main()
    print("deez")
