"""Model training"""

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from src.logger import logging
from src.transform import split_data


class BaselineModel:
    """Baseline regressor"""
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Generates the Naive forecast, which serves as the 'baseline' forecast

        Args:
            x_test (pd.DataFrame): Test set features
            features (list[str]): Column names of the features

        Returns:
            np.ndarray: 'Baseline' test set predictions
        """
        return x_test["lag_1"].values


def get_rsquared(y: pd.Series | np.ndarray, yhat: pd.Series | np.ndarray) -> float:
    """Computes the R² between y and yhat

    Args:
        y (pd.Series | np.ndarray): Targets
        yhat (pd.Series | np.ndarray): Predictions

    Returns:
        float: R²
    """
    try:
        y = y.ravel() if y.ndim > 1 else y
        yhat = yhat.ravel() if yhat.ndim > 1 else yhat
        t: pd.Series | np.ndarray = y - y.mean()
        sst: float = t.dot(t)
        e: pd.Series | np.ndarray = y - yhat
        sse: float = e.dot(e)
        return 1 - (sse / sst)
    except Exception as e:
        raise e


def train_model(
        data: pd.DataFrame, split_date: str
) -> BaselineModel | CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Trains several regressors and returns the one that produces the highest test set R²

    Args:
        data (pd.DataFrame): 2-D matrix of datetime features, window features, lag 
        features, and the target
        split_date (str): YYYY-MM-DD formatted date that's used to split data into 
        train and test sets

    Returns:
        BaselineModel | XGBRegressor | CatBoostRegressor | LGBMRegressor: Regressor that 
        produced the highest test set R²
    """
    try:
        # split the data into train and test sets
        x_train, y_train, x_test, y_test = split_data(data, split_date)

        # model dictionary
        models: dict[str, BaselineModel | XGBRegressor | CatBoostRegressor | LGBMRegressor] = {
            "BaselineModel": BaselineModel(),
            "CatBoostRegressor": CatBoostRegressor(
                loss_function="RMSE", n_estimators=100, silent=True, thread_count=-1
            ),
            "LGBMRegressor": LGBMRegressor(objective="rmse", verbose=-1, n_jobs=-1),
            "XGBRegressor": XGBRegressor(objective="reg:squarederror", n_jobs=-1)
        }

        # a dictionary to map each model to its test set R²
        report: dict[str, float] = {}
        logging.info("Model training initiated.")
        for name, model in tqdm(models.items()):
            model.fit(x_train.drop("pickup_location_id", axis=1), y_train["n_taxi_rides"])
            test_metric: float = get_rsquared(
                y_test["n_taxi_rides"], model.predict(x_test.drop("pickup_location_id", axis=1))
            )
            report[name] = round(test_metric, 4)

        # get the name of the model that produced the highest test set R²
        name_best_model: str = sorted(report.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        logging.info("Model training complete. %s selected.", name_best_model)
        return models.get(name_best_model)
    except Exception as e:
        raise e
