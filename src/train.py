"""A script that trains and evaluates select ML models"""

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import DictConfig
from tqdm import tqdm
from xgboost import XGBRegressor

from src.logger import logging
from src.paths import load_config

TRAIN_CONFIG: DictConfig = load_config().train


class NaiveForecast:
    """A class to serve as the baseline model"""

    def __str__(self):
        return "NaiveForecast"

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Returns the naive forecast

        Args:
            data (pd.DataFrame): Dataset containing the 1st lagged values.

        Returns:
            np.ndarray: Naive forecast
        """
        return data["lag_1"].values


def compute_metrics(y: pd.Series | np.ndarray, yhat: pd.Series | np.ndarray) -> dict[str, float]:
    """Computes the root mean squared error, RMSE, and coefficient of
    determination, R², between y and yhat

    Args:
        y (pd.Series | np.ndarray): Labels
        yhat (pd.Series | np.ndarray): Predictions

    Returns:
        dict[str, float]: RMSE and R²
    """
    try:
        # compute the RMSE and R² between y and yhat
        t: pd.Series | np.ndarray = y - y.mean()
        sst: float = t.dot(t)
        e: pd.Series | np.ndarray = y - yhat
        sse: float = e.dot(e)
        rmse: float = np.sqrt(sse / y.shape[0])
        r_squared: float = 1 - (sse / sst)
        return {"rmse": round(rmse, 4), "r_squared": round(r_squared, 4)}
    except Exception as e:
        raise e


def get_time_series_splits(
    data: pd.DataFrame, n_splits: int = 5
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Returns two lists, one containing the train set splits, and the other containing the
    corresponding validation set splits

    Args:
        data (pd.DataFrame): ML-ready data that the train/validation splits are based on.
        n_splits (int, optional): Number of train/validation splits to generate. Defaults to 5.

    Returns:
        tuple[list[pd.Timestamp], list[pd.Timestamp]]: Train set and validation set splits
    """
    try:
        unique_timestamps: list[pd.Timestamp] = sorted(data["pickup_time"].unique())
        horizon: int = int(round(len(unique_timestamps) / (n_splits + 1)))
        indices: list[int] = [horizon * i for i in range(1, n_splits + 1)]
        train_splits: list[pd.Timestamp] = [unique_timestamps[idx] for idx in indices]
        val_splits: list[pd.Timestamp] = [
            train_split + pd.Timedelta(hours=horizon) for train_split in train_splits
        ]
        return train_splits, val_splits
    except Exception as e:
        raise e


def train_model(
    data: pd.DataFrame, target: str = "target"
) -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Trains and evaluates select ML models and returns the one that produces
    the lowest average validation set RMSE

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features (average lag features), lag features, and the target
        target (str, optional): Column name of the target variable. Defaults to "target".

    Returns:
        CatBoostRegressor | LGBMRegressor | NaiveForecast | XGBRegressor: Trained model
        that produced the lowest average validation set RMSE
    """
    try:
        # a list containing the names of the features
        features: list[str] = [col for col in data.columns if col not in TRAIN_CONFIG.non_features]

        # a dictionary of select ML models
        models: dict[str, CatBoostRegressor | LGBMRegressor | XGBRegressor] = {
            "CatBoostRegressor": CatBoostRegressor(**TRAIN_CONFIG.CatBoostRegressor),
            "LGBMRegressor": LGBMRegressor(**TRAIN_CONFIG.LGBMRegressor),
            "XGBRegressor": XGBRegressor(**TRAIN_CONFIG.XGBRegressor)
        }

        # get the train set and validation set splits
        train_splits, val_splits = get_time_series_splits(data)

        # an empty dictionary to map each trained model to its average validation set RMSE
        report: dict[str, float] = {}

        # for each model ...
        for name, model in tqdm(models.items()):
            logging.info("Training initiated for the %s.", name)
            # create an empty list to store its evaluation metrics, one per train/validation split
            eval_metrics: list[float] = []

            # iterate over each split, train it, compute its validation set RMSE, and ...
            # save it to the 'eval_metrics' list
            for train_split, val_split in zip(train_splits, val_splits):
                train_query: str = f"pickup_time >= '{train_split}'"
                val_query: str = f"pickup_time < '{val_split}'"
                x_train: pd.DataFrame = data.query(train_query)[features]
                y_train: pd.Series = data.query(train_query)[target]
                x_val: pd.DataFrame = data.query(val_query)[features]
                y_val: pd.Series = data.query(val_query)[target]

                # if the model is an object of type, 'CatBoostRegressor'
                if isinstance(model, CatBoostRegressor):
                    model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                    metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

                # if the model is an object of type, 'LGBMRegressor'
                elif isinstance(model, LGBMRegressor):
                    model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
                    metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

                # if the model is an object of type, 'XGBRegressor'
                else:
                    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                    metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

                # append the split's validation set RMSE to the 'eval_metrics' list
                eval_metrics.append(metric)

            # replace the untrained model in the 'models' dictionary with its trained version
            models[name] = model

            # map the trained model to its average validation set RMSE
            report[name] = np.mean(eval_metrics)

        # get the trained model that produced the lowest average validation set RMSE
        best_model: str = (
            pd.DataFrame.from_dict(report, orient="index", columns=["rmse"])
            .sort_values("rmse")
            .index[0]
        )
        logging.info(
            "Training complete, the %s produced the lowest average validation set RMSE.", best_model
        )
        return models.get(best_model)
    except Exception as e:
        raise e