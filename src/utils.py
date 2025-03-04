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


@logger.catch
def transform_data(
    data: pl.DataFrame,
    target_col: str = data_config.target_column,
    temporal_col: str = data_config.temporal_column,
    max_lag: int = 24
) -> pl.DataFrame:
    """Transforms pre-processed and validated hourly taxi rides data into an ML-ready
    dataset that contains lag features, average lag features, datetime features, and
    the corresponding target.

    Args:
        data (pl.DataFrame): DataFrame containing a 1-D time series of pre-processed and
        validated taxi rides data.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_col (str, opitonal): Name of the column that contains the datetime objects.
        Defaults to data_config.temporal_column.
        max_lag (int, optional): Maximum number of lag features to create. Defaults to 24.

    Returns:
        pl.DataFrame: ML-ready dataset that contains lag features, average lag features,
        datetime features, and the corresponding target.
    """
    try:
        logger.info("Transforming the pre-processed and validated data into features and targets.")
        transformed_dfs: list[pl.DataFrame] = []
        for location_id in tqdm(sorted(data["location_id"].unique()), unit="Location ID"):
            # create the lag features
            dfs: list[pl.DataFrame] = [
                (
                    data
                    .filter(pl.col("location_id").eq(location_id))
                    .select(target_col)
                    .shift(n=lag)
                    .rename({target_col: f"lag_{lag}"})
                )
                for lag in reversed(range(1, max_lag + 1))
            ]
            df_lags: pl.DataFrame = pl.concat(dfs, how="horizontal").drop_nulls()

            # create the average lag features, i.e., window features
            start = step = 4
            dfs: list[pl.DataFrame] = [
                (
                    df_lags[:, -lag:]
                    .mean_horizontal()
                    .to_frame(name=f"avg_{lag}_lags")
                )
                for lag in reversed(range(start, max_lag + 1, step))
            ]
            df_avg_lags: pl.DataFrame = pl.concat(dfs, how="horizontal")

            # create the datetime features
            # NOTE: this pl.DataFrame also includes the 'location_id' and 'temporal_col' columns
            df_datetime: pl.DataFrame = (
                data
                .filter(pl.col("location_id").eq(location_id))
                .select(["location_id", temporal_col])
                .shift(n=-max_lag)
                .drop_nulls()
                .with_columns(
                    (
                        pl.col(temporal_col)
                        .dt.convert_time_zone(time_zone="UTC")
                        .dt.convert_time_zone(time_zone="EST")
                        .dt.hour()
                        .cast(pl.Int32)
                        .alias("hour")
                    )
                )
                .with_columns(
                    pl.when(pl.col("hour").ge(5) & pl.col("hour").lt(12)).then(1)  # morning
                    .when(pl.col("hour").ge(12) & pl.col("hour").lt(17)).then(2)  # afternoon
                    .when(pl.col("hour").ge(17) & pl.col("hour").lt(21)).then(3)  # evening
                    .otherwise(4)  # night
                    .alias("time_of_day")
                )
                .select(["location_id", temporal_col, "hour", "time_of_day"])
            )

            # horizontally concatenate the features and add the corresponding target
            transformed_data: pl.DataFrame = (
                pl.concat((df_datetime, df_avg_lags, df_lags), how="horizontal")
                .join(data, how="left", on=["location_id", temporal_col])
            )
            transformed_dfs.append(transformed_data)
        return pl.concat(transformed_dfs, how="vertical")
    except Exception as e:
        raise e
