"""Streamlit web application"""

from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
import streamlit as st

from plotly.graph_objects import Figure
from pydeck import Deck, Layer, ViewState

from src.inference import generate_forecast
from src.ingest import load_taxi_zones
from src.transform import fetch_and_transform, plot_record

N_STEPS: int = 4


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


# set the page layout
st.set_page_config(layout="wide")

# set the title and header
current_date: pd.Timestamp = pd.Timestamp(datetime.now(timezone.utc))
st.title("Taxi demand forecasting :taxi:")
st.header(f"{current_date.replace(microsecond=0, tzinfo=None)} UTC")

# create a side-of-the-page progress bar
progress_bar = st.sidebar.header(":gear: Progress Bar")
progress_bar = st.sidebar.progress(0)

# fetch the latest taxi demand data and transform it into ML-ready features and labels
with st.spinner("Fetching and transforming the latest taxi demand data"):
    df: pd.DataFrame = fetch_and_transform()
    st.sidebar.write(":white_check_mark: Data fetched and transformed into features and labels")
    progress_bar.progress(1 / N_STEPS)
    # print(df.query("location_id == 43").iloc[-1:, :])
    
# generate the one-step forecast, i.e., the predicted taxi demand for the upcoming hour
with st.spinner("Forecasting taxi demand for the upcoming hour"):
    forecast: pd.DataFrame = (
        df
        .pipe(generate_forecast)
        .pipe(color_code_forecasts)
        .sort_values("forecast", ascending=False)
        .reset_index(drop=True)
    )
    st.sidebar.write(":white_check_mark: Taxi demand forecasted")
    progress_bar.progress(2 / N_STEPS)
    # print(forecast.query("location_id == 43"))

# load the shape files containing the NYC taxi zones data
with st.spinner("Downloading the taxi zone shape files"):
    gdf: gpd.GeoDataFrame = load_taxi_zones().merge(forecast, on="location_id")
    st.sidebar.write(":white_check_mark: Taxi zone shapefiles downloaded")
    progress_bar.progress(3 / N_STEPS)
    print(gdf.query("location_id == 43"))

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
    for location_id in forecast["location_id"].head(10):
        fig: Figure = plot_record(forecast, location_id)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
    st.sidebar.write(":white_check_mark: Map and time-series plots generated")
    progress_bar.progress(4 / N_STEPS)
