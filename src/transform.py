"""Data transformation"""

import pandas as pd

from tqdm import tqdm

from src.logger import logging


def add_datetime_features(data: pd.DataFrame, target_name: str = "n_taxi_rides") -> pd.DataFrame:
    """Adds datetime features to a matrix that already contains lag features and the target

    Args:
        data (pd.DataFrame): 2-D matrix that contains lag features and the target.
        target_name (str, optional): Column name of the target. Defaults to "n_taxi_rides".

    Returns:
        pd.DataFrame: 2-D matrix of datetime features, lag features, and the target
    """
    try:
        output_cols: list[str] = (
            ["pickup_location_id", "hour", "time_of_day"] +
            [col for col in data.columns if col.startswith("lag")] +
            [target_name]
        )
        return (
            data
            .assign(
                hour=data.index.hour,
                time_of_day=[
                    1 if hour in range(5, 12)
                    else 2 if hour in range(12, 17)
                    else 3 if hour in range(17, 21)
                    else 4 for hour in data.index.hour
                ]
            )
            [output_cols]
        )
    except Exception as e:
        raise e


def add_window_features(data: pd.DataFrame, target_name: str = "n_taxi_rides") -> pd.DataFrame:
    """Adds window features, that is, the average across multiple lag features, to a matrix 
    that already contains datetime features, lag features, and the target

    Args:
        data (pd.DataFrame): 2-D matrix that contains datetime features, lag features, and 
        the target.
        target_name (str, optional): Column name of the target. Defaults to "n_taxi_rides".

    Returns:
        pd.DataFrame: 2-D matrix of datetime features, window features, lag features, and 
        the target
    """
    try:
        lag_cols: list[str] = [col for col in data.columns if col.startswith("lag")]
        avg_lags: list[pd.Series] = [
            data[lag_cols[i:]].mean(axis=1) for i in range(-len(lag_cols), 0, 4)
        ]
        df: pd.DataFrame = pd.concat(avg_lags, axis=1)
        df.columns = [f"mean_lags_{i}_to_1" for i in reversed(range(4, len(lag_cols) + 1, 4))]
        output_cols: list[str] = (
            ["pickup_location_id", "hour", "time_of_day"] +
            df.columns.tolist() + lag_cols + [target_name]
        )
        return pd.concat((data, df), axis=1)[output_cols]
    except Exception as e:
        raise e


def convert_series_to_matrix(
        data: pd.DataFrame, pickup_location_id: int, n_lags: int = 24
) -> pd.DataFrame:
    """Transforms a univariate time series into a 2-D matrix of datetime features, lag 
    features, and the target

    Args:
        data (pd.DataFrame): Pre-processed and validated data that contains several 
        univariate time series, one per location ID.
        pickup_location_id (int): Location ID for the univariate time series of interest.
        n_lags (int, optional): Number of lag features to create. Defaults to 24.

    Returns:
        pd.DataFrame: 2-D matrix of datetime features, lag features, and the target
    """
    try:
        time_series: pd.Series = (
            data
            .query(f"pickup_location_id == {pickup_location_id}")
            .sort_values("pickup_hour")
            .set_index("pickup_hour")
            ["n_taxi_rides"]
        )
        lags: list[pd.Series] = [
            time_series.shift(periods=lag) for lag in reversed(range(1, n_lags + 1))
        ]
        lag_cols: list[str] = [f"lag_{lag}" for lag in reversed(range(1, n_lags + 1))]
        matrix: pd.DataFrame = pd.concat(lags, axis=1).dropna()
        matrix.columns = lag_cols
        matrix = (
            matrix
            .assign(
                pickup_location_id=pickup_location_id,
                n_taxi_rides=time_series.loc[matrix.index]
            )
            [["pickup_location_id"] + lag_cols + ["n_taxi_rides"]]
        )
        return matrix.pipe(add_datetime_features).pipe(add_window_features)
    except Exception as e:
        raise e


def transform_data(data: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
    """Iterates over each unique pickup location ID and transforms its corresponding 
    univariate time series into a 2-D matrix of datetime features, lag features, and 
    the target.

    Args:
        data (pd.DataFrame): Pre-processed and validated data that contains several 
        univariate time series, one per pickup location ID.
        n_lags (int, optional): Number of lag features to create. Defaults to 24. 

    Returns:
        pd.DataFrame: 2-D matrix of datetime features, lag features, the target, and 
        pickup location ID, for which the datetime features, lag features, and target 
        correspond to
    """
    try:
        logging.info("Data transformation initiated.")
        matrices: list[pd.DataFrame] = [
            convert_series_to_matrix(data, pickup_location_id, n_lags)
            for pickup_location_id in tqdm(sorted(data["pickup_location_id"].unique()))
        ]
        logging.info("Data transformation complete.")
        return pd.concat(matrices, axis=0)
    except Exception as e:
        raise e


def split_data(
        data: pd.DataFrame, split_date: str, target_name: str = "n_taxi_rides"
) -> tuple[pd.DataFrame, ...]:
    """Splits the transformed data into train and test sets

    Args:
        data (pd.DataFrame): 2-D matrix of datetime features, window features, lag 
        features, and the target
        split_date (str): YYYY-MM-DD formatted date with which to split data into 
        train and test sets
        target_name (str, optional): Column name of the target. Defaults to "n_taxi_rides".

    Returns:
        tuple[pd.DataFrame, ...]: Train and test set features and targets
    """
    try:
        logging.info("Splitting the transformed data into train and test sets.")
        train_data: pd.DataFrame = data.loc[data.index < split_date]
        test_data: pd.DataFrame = data.loc[data.index >= split_date]
        x_train: pd.DataFrame = train_data.drop(target_name, axis=1)
        y_train: pd.DataFrame = train_data[["pickup_location_id", target_name]]
        x_test: pd.DataFrame = test_data.drop(target_name, axis=1)
        y_test: pd.DataFrame = test_data[["pickup_location_id", target_name]]
        return x_train, y_train, x_test, y_test
    except Exception as e:
        raise e
