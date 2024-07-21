"""A script that downloads, validates, and pre-processes raw NYC taxi rides data"""

import calendar
import os

from pathlib import Path, PosixPath
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import polars as pl
import requests

from omegaconf import OmegaConf
from requests import Response
from tqdm import tqdm

from src.config import Config, load_config
from src.logger import logging

INGEST_CONFIG: dict[str, list[str]] = OmegaConf.to_container(load_config().ingest)


def validate_data(data: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Returns a pd.DataFrame containing only records (taxi rides) for the
    specified year and month, that's also free of null values and duplicates

    Args:
        data (pd.DataFrame): Raw data
        year (int): Raw data's recorded year
        month (int): Raw data's recorded month

    Returns:
        pd.DataFrame: Validated data
    """
    try:
        start: pd.Timestamp = pd.Timestamp(f"{year}-{month:02d}-01")
        last_day: int = calendar.monthrange(year, month)[1]
        end: pd.Timestamp = pd.Timestamp(f"{year}-{month:02d}-{last_day}") + pd.Timedelta(days=1)
        return (
            data.assign(pickup_datetime=pd.to_datetime(data["tpep_pickup_datetime"]))[
                (pd.to_datetime(data["tpep_pickup_datetime"]) >= start)
                & (pd.to_datetime(data["tpep_pickup_datetime"]) < end)
            ]
            .dropna(subset=["tpep_pickup_datetime", "PULocationID"])
            .drop_duplicates(subset=["tpep_pickup_datetime", "PULocationID"], keep="first")
            .rename({"PULocationID": "location_id"}, axis=1)
            .sort_values(["location_id", "pickup_datetime"])
            .reset_index(drop=True)[INGEST_CONFIG.get("raw_columns")]
        )
    except Exception as e:
        raise e


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Returns a pd.DataFrame containing taxi rides recorded at
    regularly spaced hourly timestamps, for each unique location ID

    Args:
        data (pd.DataFrame): Validated data

    Returns:
        pd.DataFrame: Pre-rocessed data
    """
    try:
        # a list of pre-processed pd.DataFrames, one per location ID
        dfs: list[pd.DataFrame] = [
            (
                data.query(f"location_id == {location_id}")
                .assign(pickup_datetime=data["pickup_datetime"].dt.floor("H"))
                .groupby("pickup_datetime", as_index=False)
                .agg(rides=("pickup_datetime", "size"))
                .set_index("pickup_datetime")
                .asfreq("H")
                .fillna(0)
                .assign(location_id=location_id)
                .reset_index()[INGEST_CONFIG.get("processed_columns")]
            )
            for location_id in tqdm(data["location_id"].unique())
        ]
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e


def download_file(year: int, month: int) -> pd.DataFrame:
    """Downloads a single file of raw data from the 'NYC trip data' URL,
    validates and pre-processes it, and returns a pd.DataFrame containing
    taxi rides recorded at an hourly frequency, for each location ID

    Args:
        year (int): Raw data's recorded year
        month (int): Raw data's recorded month
    """
    try:
        filename: str = f"yellow_tripdata_{year}-{month:02d}.parquet"
        response: Response = requests.get(os.path.join(Config.RAW_DATA_URL, filename))
        if response.status_code == 200:
            logging.info(
                "Downloading, validating, and pre-processing %s",
                os.path.join(Config.RAW_DATA_URL, filename)
            )
            return (
                pd.read_parquet(os.path.join(Config.RAW_DATA_URL, filename))
                .pipe(validate_data, year, month)
                .pipe(preprocess_data)
            )
        else:
            logging.info("%s is not available", os.path.join(Config.RAW_DATA_URL, filename))
    except Exception as e:
        raise e


def fetch_and_preprocess(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetches raw data, then validates, pre-processes, filters, and returns 
    it as a pd.DataFrame containing NYC taxi rides, recorded at regularly 
    sampled milliseconds from the Unix epoch 

    Args:
        start (pd.Timestamp): Starting timestamp used to filter the data
        end (pd.Timestamp): Ending timestamp used to filter the data

    Returns:
        pd.DataFrame: NYC taxi demand data, recorded at regularly sampled 
        milliseconds from the Unix epoch
    """
    try:
        # a list of validated and pre-processed pd.DataFrames, one for each (year, month) pair
        dfs: list[pd.DataFrame] = [
            download_file(year, month)
            for year, month in zip([start.year, end.year], [start.month, end.month])
        ]
        return (
            pl.concat([pl.from_pandas(df) for df in dfs], how="vertical")
            .filter(
                pl.col("pickup_datetime").dt.replace_time_zone("UTC").ge(start),
                pl.col("pickup_datetime").dt.replace_time_zone("UTC").le(end),
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

    
def load_taxi_zones() -> gpd.GeoDataFrame:
    """Downloads a zip file, unzips its contents (shapefiles of NYC taxi zones), 
    reads in and returns the shapefile as a gpd.GeoDataFrame

    Returns:
        gpd.GeoDataFrame: Dataset containing geographic information about NYC taxi zones
    """
    try:
        response: Response = requests.get(Config.SHAPEFILES_URL)
        if response.status_code == 200:
            # create the 'data' sub-directory, ~/data, if it doesn't already exist
            data_dir: PosixPath = Config.DATA_DIR
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # get the name of the zip file, which is 'taxi_zones.zip'
            zip_file: str = Path(Config.SHAPEFILES_URL).name
            
            # download the zip file to ~/data/taxi_zones.zip
            open(data_dir / zip_file, "wb").write(response.content)
            
            # upzip the zip file and save its contents (shape files) to ~/data/taxi_zones/
            ZipFile(data_dir / zip_file, "r").extractall(data_dir / zip_file.replace(".zip", ""))
            
            # remove the zip file, ~/data/taxi_zones.zip
            os.remove(data_dir / zip_file)
            
            # read in ~/data/taxi_zones/taxi_zones.shp as a gpd.GeoDataFrame
            output_cols: list[str] = [
                "object_id", 
                "shape_length", 
                "shape_area", 
                "zone", 
                "location_id", 
                "borough", 
                "geometry"
            ]
            gdf: gpd.GeoDataFrame = gpd.read_file(
                data_dir / zip_file.replace(".zip", "") / zip_file.replace("zip", "shp")
            )
            return gdf.rename(dict(zip(gdf.columns, output_cols)), axis=1).to_crs("epsg: 4326")
        else:
            logging.info("%s is not available", Config.SHAPEFILES_URL)
    except Exception as e:
        raise e
