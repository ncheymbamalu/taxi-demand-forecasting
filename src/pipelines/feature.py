"""This module provides functionality for executing the feature pipeline."""

from datetime import datetime, timedelta, timezone

import polars as pl

from src.config import Paths, data_config
from src.utils import fetch_data


def main() -> None:
    """Fetches raw data from the NYC taxi data API, then pre-processes, validates, and
    concatenates it with previously processed data. Finally, Paths.DATA is updated with
    the concatenated data.
    """
    try:
        temporal_col: str = data_config.temporal_column
        end: datetime = (
            datetime
            .now(timezone.utc)
            .replace(
                minute=0,
                second=0,
                microsecond=0,
                tzinfo=None
            )
            - timedelta(days=366)
        )
        start: datetime = end - timedelta(days=data_config.duration)
        months: list[int] = [
            12 if end.month == 1 else end.month - 1,
            end.month
        ]
        years: list[int] = [
            end.year - 1 if months[0] == 12 else end.year,
            end.year
        ]
        dfs: list[pl.DataFrame] = [fetch_data(year, month) for year, month in zip(years, months)]
        data: pl.DataFrame = (
            pl.concat(dfs, how="vertical")
            .filter(
                pl.col(temporal_col).ge(start)
                & pl.col(temporal_col).le(end)
            )
            .with_columns(
                pl.col(temporal_col) + timedelta(days=366)
            )
        )
        (
            pl.concat((pl.read_parquet(Paths.DATA), data), how="vertical")
            .sort(by=["location_id", temporal_col])
            .unique(subset=["location_id", temporal_col], keep="first", maintain_order=True)
            .write_parquet(Paths.DATA)
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
