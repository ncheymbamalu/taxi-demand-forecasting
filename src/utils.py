"""This module contains helper functions that are used in ~/src/app.py."""

import os

from pathlib import Path, PosixPath
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import plotly.express as px
import requests

from plotly.graph_objects import Figure
from requests import Response

from src.config import Paths, load_config
from src.logger import logger


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
        # get the lag features
        lag_features: list[str] = [col for col in data.columns if col.startswith("lag")]

        # get the record's timestamps
        end: pd.Timestamp = data.query(f"location_id == {location_id}").squeeze()["pickup_time"]
        start: pd.Timestamp = end - pd.Timedelta(hours=len(lag_features))
        timestamps: list[pd.Timestamp] = pd.date_range(start, end, freq="H").to_list()

        # instantiate an object of type, 'Figure', with the lag features
        fig: Figure = px.line(
            x=timestamps[:-1],
            y=data.query(f"location_id == {location_id}")[lag_features].squeeze(),
            color_discrete_sequence=["blue"],
            labels={"x": "Datetime (UTC)", "y": "Number of taxi rides"},
            template="plotly_dark",
            markers=True,
            title=f"Location ID: {location_id}, Pick-Up Time: {end}"
        )

        # add the target to the 'Figure' instance
        fig.add_scatter(
            x=[timestamps[-1]],
            y=(
                data.query(f"location_id == {location_id}")["target"].tolist()
                if "target" in data.columns
                else data.query(f"location_id == {location_id}")["forecast"].tolist()
            ),
            line_color="green",
            mode="markers",
            marker_size=10,
            name="Target" if "target" in data.columns else "Forecast"
        )

        # add the forecast to the 'Figure' instance
        if plot_forecast:
            fig.add_scatter(
                x=[timestamps[-1]],
                y=data.query(f"location_id == {location_id}")["forecast"].tolist(),
                line_color="red",
                mode="markers",
                marker_size=10,
                name="Forecast"
            )
        return fig
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
        corresponding fill colors
    """
    try:
        normalized_forecasts: pd.Series = (
            (data["forecast"] - data["forecast"].min()) /
            (data["forecast"].max() - data["forecast"].min())
        )
        rgb_greens: list[tuple[int, ...]] = [
            tuple((0, int(round(forecast * 255)), 0)) for forecast in normalized_forecasts
        ]
        return data.assign(fill_color=rgb_greens)
    except Exception as e:
        raise e


def download_taxi_zones() -> None:
    """Downloads a zip file whose contents are shapefiles of NYC taxi zones, unzips
    the contents, and saves them to ~/data/taxi_zones/
    """
    try:
        response: Response = requests.get(Paths.TAXI_ZONES_SHAPEFILES_URL)
        if response.status_code == 200:
            # create ~/data/ if it doesn't already exist
            data_dir: PosixPath = Paths.DATA_DIR
            data_dir.mkdir(parents=True, exist_ok=True)

            # get the url's base name, which is 'taxi_zones.zip'
            base_name: str = Path(Paths.TAXI_ZONES_SHAPEFILES_URL).name

            # save the url's contents to ~/data/taxi_zones.zip
            open(data_dir / base_name, "wb").write(response.content)

            # unzip ~/data/taxi_zones.zip and save its contents (shapefiles) to ~/data/taxi_zones/
            ZipFile(data_dir / base_name, "r").extractall(data_dir / base_name.replace(".zip", ""))

            # delete ~/data/taxi_zones.zip
            os.remove(data_dir / base_name)
        else:
            logger.info(f"Invalid request. {Paths.TAXI_ZONES_SHAPEFILES_URL} is not available.")
    except Exception as e:
        raise e


def read_taxi_zones() -> gpd.GeoDataFrame:
    """Reads in ~/data/taxi_zones/taxi_zones.shp and returns a GeoDataFrame

    Returns:
        gpd.GeoDataFrame: Dataset containing geographic information about NYC taxi zones
    """
    try:
        shapefile: PosixPath = Paths.DATA_DIR / "taxi_zones" / "taxi_zones.shp"
        if os.path.exists(shapefile):
            gdf: gpd.GeoDataFrame = gpd.read_file(shapefile)
        else:
            download_taxi_zones()
            gdf: gpd.GeoDataFrame = gpd.read_file(shapefile)
        return (
            gdf
            .to_crs("epsg: 4326")
            .rename(dict(zip(gdf.columns, load_config().utils.taxi_zone_columns)), axis=1)
            .sort_values(by="location_id")
            .reset_index(drop=True)
            [["location_id", "zone", "geometry"]]
        )
    except Exception as e:
        raise e
