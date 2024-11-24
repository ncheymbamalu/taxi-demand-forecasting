"""This module generates the Streamlit web application."""

from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
import streamlit as st

from plotly.graph_objects import Figure
from pydeck import Deck, Layer, ViewState

from src.inference import generate_forecast
from src.transform import fetch_and_transform
from src.utils import color_code_forecasts, plot_record, read_taxi_zones

N_STEPS: int = 4

# set the page layout
st.set_page_config(layout="wide")

# set the title and header
current_time: pd.Timestamp = (
    pd.Timestamp(datetime.now(timezone.utc)).replace(microsecond=0, tzinfo=None)
)
st.title("NYC taxi demand forecasting :taxi:")
st.header(f"{current_time} UTC")

# create a side-of-the-page progress bar
progress_bar = st.sidebar.header(":gear: Progress Bar")
progress_bar = st.sidebar.progress(0)

# fetch the latest data from Hopsworks and transform it into ML-ready features and labels
with st.spinner("Fetching and transforming the latest taxi demand data"):
    df: pd.DataFrame = fetch_and_transform()
    st.sidebar.write(":white_check_mark: Data fetched and transformed into features and labels")
    progress_bar.progress(1 / N_STEPS)

# generate each location's one-step forecast, i.e., the predicted taxi demand for the upcoming hour
with st.spinner("Forecasting taxi demand for the upcoming hour"):
    df_forecast: pd.DataFrame = (
        df
        .pipe(generate_forecast)
        .pipe(color_code_forecasts)
        .sort_values("forecast", ascending=False)
        .reset_index(drop=True)
    )
    st.sidebar.write(":white_check_mark: Taxi demand forecasted")
    progress_bar.progress(2 / N_STEPS)

# load the NYC taxi zones data and merge with the forecasted data
with st.spinner("Loading the taxi zone shapefiles"):
    gdf: gpd.GeoDataFrame = read_taxi_zones().merge(df_forecast, how="right", on="location_id")
    st.sidebar.write(":white_check_mark: Taxi zone shapefiles loaded")
    progress_bar.progress(3 / N_STEPS)

# create the visualizations
with st.spinner("Generating visualizations"):
    # create a map of NYC
    initial_view_state: ViewState = ViewState(
        longitude=-73.935242, latitude=40.73061, zoom=10, max_zoom=15, pitch=45, bearing=0
    )
    layer: Layer = Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=(255, 255, 255),
        auto_highlight=True,
        pickable=True
    )
    tooltip: dict[str, str] = {
        "html": """
        <b>Location ID</b>: {location_id}<br/>
        <b>Zone</b>: {zone}<br/>
        <b>Forecasted Taxi Demand</b>: {forecast}
        """
    }
    st.pydeck_chart(Deck(layers=[layer], initial_view_state=initial_view_state, tooltip=tooltip))

    # create a scatter plot for the 10 busiest location IDs, based on forecasted taxi demand
    for location_id in df_forecast["location_id"].head(10):
        fig: Figure = plot_record(df_forecast, location_id)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
    st.sidebar.write(":white_check_mark: Map and time-series plots generated")
    progress_bar.progress(4 / N_STEPS)
