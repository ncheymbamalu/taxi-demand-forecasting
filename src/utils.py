"""A script that contains helper functions for ~/src/app.py"""

import os
import random

from pathlib import Path, PosixPath
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import plotly.express as px
import requests

from plotly.graph_objects import Figure
from requests import Response

from src.config import Config
from src.ingest import INGEST_CONFIG
from src.logger import logging



def plot_record(data: pd.DataFrame, location_id: int, plot_forecast: bool = False) -> Figure:
    """Plots a single record (row) of lag features, the corresponding target, and 
    forecast, if the 'plot_forecast' parameter is set equal to True

    Args:
        data (pd.DataFrame): Dataset containing, at mininum, lag features and the
        target. The forecast is optional.
        location_id (int): The record's location ID
        plot_forecast (bool): Boolean that determines if the forecast is plotted. 
        Defaults to False.
    """
    try:
        # inputs to the 'fig' object
        idx: int = random.choice(data.query(f"location_id == {location_id}").index)
        end: pd.Timestamp = data.loc[idx, "pickup_datetime"]
        lag_cols: list[str] = [col for col in data.columns if col.startswith("lag")]
        start: pd.Timestamp = end - pd.Timedelta(hours=len(lag_cols))
        timestamps: pd.DatetimeIndex = pd.date_range(start, end, freq="H")

        # instantiate the 'fig' object with the lag features
        fig: Figure = px.line(
            x=timestamps[:-1],
            y=data.loc[idx, lag_cols],
            color_discrete_sequence=["blue"],
            labels={"x": "Datetime (UTC)", "y": "Number of taxi rides"},
            template="plotly_dark",
            markers=True,
            title=f"Location ID: {location_id}, Pick-Up Time: {end}",
        )

        # add the target to the 'fig' object
        fig.add_scatter(
            x=[timestamps[-1]],
            y=(
                [data.loc[idx, "target"]]
                if "target" in data.columns else [data.loc[idx, "forecast"]]
            ),
            line_color="green",
            mode="markers",
            marker_size=10,
            name="Target" if "target" in data.columns else "Forecast",
        )
        
        if plot_forecast:
            # add the forecast to the 'fig' object
            fig.add_scatter(
                x=[timestamps[-1]],
                y=[data.loc[idx, "forecast"]],
                line_color="red",
                mode="markers",
                marker_size=10,
                name="Forecast"
            )
        return fig
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
            gdf: gpd.GeoDataFrame = gpd.read_file(
                data_dir / zip_file.replace(".zip", "") / zip_file.replace("zip", "shp")
            )
            return (
                gdf
                .rename(dict(zip(gdf.columns, INGEST_CONFIG.get("taxi_zone_columns"))), axis=1)
                .to_crs("epsg: 4326")
                [["location_id", "zone", "geometry"]]
            )
        else:
            logging.info("%s is not available", Config.SHAPEFILES_URL)
    except Exception as e:
        raise e


def color_code_forecasts(data: pd.DataFrame) -> pd.DataFrame:
    """Color codes the forecasted taxi demand with different shades of 
    green, where lighter shades correspond to a larger forecasted demand 
    and darker shades correspond to a smaller forecasted demand

    Args:
        data (pd.DataFrame): Dataset containing the forecasted taxi demand

    Returns:
        pd.DataFrame: Dataset containing the forecasted taxi demand and the
        corresponding RGB colors
    """
    try:
        normalized_forecasts: list[float] = [
            (forecast - data["forecast"].min()) / 
            (data["forecast"].max() - data["forecast"].min())
            for forecast in data["forecast"]
        ]
        rgb_colors: list[tuple[int, ...]] = [
            tuple((0, int(round(normalized_forecast * 255)), 0))
            for normalized_forecast in normalized_forecasts
        ]
        return data.assign(fill_color=rgb_colors)
    except Exception as e:
        raise e
