"""This function contains utility/helper functions."""

import os

import httpx
import polars as pl

from httpx import Response
from tqdm import tqdm

from src.config import Paths, data_config
from src.logger import logger


@logger.catch
def fetch_data(year: int, month: int) -> pl.DataFrame:
    """Fetches raw data from the NYC taxi data API, then pre-processes, validates, and
    returns it as a pl.DatFrame.

    Args:
        year (int): Raw data's recorded year.
        month (int): Raw data's recorded month.

    Returns:
        pl.DataFrame: Pre-processed and validated data that contains hourly NYC taxi rides.
    """
    try:
        path: str = os.path.join(Paths.TAXI_DATA_API, f"yellow_tripdata_{year}-{month:02d}.parquet")
        response: Response = httpx.get(url=path)
        if response.status_code == 200:
            logger.info(f"Downloading, pre-processing, and validating raw data from {path}.")
            data: pl.DataFrame = pl.read_parquet(path)
            dfs: list[pl.DataFrame] = [
                (
                    data
                    .filter(pl.col("PULocationID").eq(location_id))
                    .sort(by="tpep_pickup_datetime")
                    .with_columns(
                        pl.col("tpep_pickup_datetime").dt.truncate("1h")
                    )
                    .group_by("tpep_pickup_datetime")
                    .count()
                    .upsample(
                        time_column="tpep_pickup_datetime",
                        every="1h",
                        maintain_order=True
                    )
                    .fill_null(0)
                    .with_columns(
                        pl.col("count").cast(pl.Int32),
                        location_id=location_id
                    )
                    .rename({
                        "tpep_pickup_datetime": "pickup_time",
                        "count": "n_rides"
                    })
                    .select(data_config.columns)
                )
                for location_id in tqdm(sorted(data["PULocationID"].unique()), unit="Location ID")
            ]
            data = (
                pl.concat(dfs, how="vertical")
                .unique(maintain_order=True, keep="first")
            )
            assert data.is_duplicated().sum() == 0
            assert data.null_count().sum_horizontal()[0] == 0
            return data
        logger.info(f"Invalid request. {path} is not available to download.")
        return pl.DataFrame(schema=data_config.columns)
    except Exception as e:
        raise e
