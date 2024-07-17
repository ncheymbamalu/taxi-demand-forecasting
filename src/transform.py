"""A script to transform a 1-D time series into a tabular, ML-ready dataset"""

import random

import pandas as pd
import plotly.express as px

from plotly.graph_objects import Figure
from tqdm import tqdm

from src.logger import logging


def tabularize_data(data: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    """Converts a pd.DataFrame containing a 1-D time series of validated and
    pre-processed hourly taxi rides to a pd.DataFrame of lag features, window
    features (average lag features), and the corresponding target

    Args:
        data (pd.DataFrame): Validated and pre-processed data
        max_lag (int, optional): Maximum number of lag features to create. Defaults to 24.

    Returns:
        pd.DataFrame: Tabular, ML-ready dataset containing lag features, window features,
        and the corresponding target
    """
    try:
        logging.info("Transforming the 1-D time series into a tabular, ML-ready dataset.")
        # an empty list to store the tabular pd.DataFrames, one per location ID
        dfs: list[pd.DataFrame] = []
        for location_id in tqdm(data["location_id"].unique()):
            # extract the lag features
            lags: list[pd.DataFrame] = [
                (
                    data
                    .query(f"location_id == {location_id}")
                    .drop("location_id", axis=1)
                    .set_index("pickup_datetime")
                    .shift(periods=lag)
                    .rename({"rides": f"lag_{lag}"}, axis=1)
                )
                for lag in reversed(range(1, max_lag + 1))
            ]

            # extract the window features, i.e., average lag features
            avg_lags: list[pd.DataFrame] = [
                (
                    pd.concat(lags, axis=1)
                    .dropna()
                    .iloc[:, -lag:].mean(axis=1)
                    .to_frame()
                    .rename({0: f"avg_{lag}_lags"}, axis=1)
                )
                for lag in reversed(range(4, max_lag + 1, 4))
            ]

            # horizontally concatenate the window features and lag features
            tabular_data: pd.DataFrame = pd.concat(avg_lags + lags, axis=1).dropna()

            # a list of all the feature names
            features: list[str] = (
                ["hour", "day_of_week"]
                + [col for col in tabular_data.columns if col.startswith("avg")]
                + [col for col in tabular_data.columns if col.startswith("lag")]
            )
            
            # final processing 
            # (1) add two datetime features, based on the 'pickup_datetime' index, ...
            # one that extracts the hour and one that extracts the day of the week 
            # (2) add the location ID
            # (3) add the corresponding target
            # (4) reset the index so that the 'pickup_datetime' index becomes a column
            # (5) re-arrange the resulting columns
            tabular_data = (
                tabular_data
                .assign(
                    hour=tabular_data.index.hour,
                    day_of_week=tabular_data.index.day_of_week,
                    location_id=location_id,
                    target=(
                        data
                        .query(f"location_id == {location_id}")
                        .set_index("pickup_datetime")
                        .loc[tabular_data.index, "rides"]
                    ),
                )
                .reset_index()
                [["location_id", "pickup_datetime"] + features + ["target"]]
            )
            dfs.append(tabular_data)
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e
    

def plot_record(data: pd.DataFrame) -> None:
    """Plots a single row of lag features and the corresponding target

    Args:
        data (pd.DataFrame): Tabular, ML-ready dataset
    """
    try:
        # inputs to the 'fig' object
        location_id: int = random.choice(data["location_id"].unique().tolist())
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
            labels={"x": "Datetime", "y": "Number of taxi rides"},
            template="plotly_dark", 
            markers=True,
            title=f"Location id: {location_id}, Day and time of pick up: {end}"
        )
        
        # add the target to the 'fig' object
        fig.add_scatter(
            x=[timestamps[-1]], 
            y=[data.loc[idx, "target"]], 
            line_color="green",
            mode="markers",
            marker_size=10,
            name="target"
        )
        fig.show()
    except Exception as e:
        raise e
