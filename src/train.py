"""A script that trains, optimizes, and evaluates select machine learning models"""

import numpy as np
import optuna
import pandas as pd

from lightgbm import LGBMRegressor
from omegaconf import OmegaConf
from optuna import Study, Trial
from tqdm import tqdm
from xgboost import XGBRegressor

from src.config import Config, load_config

TRAIN_CONFIG: dict[str, dict[str, int | str]] = OmegaConf.to_container(load_config().train)


class NaiveForecast:
    """A class to serve as the baseline model"""

    def __str__(self):
        return "NaiveForecast"

    def predict(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Returns the naive forecast

        Args:
            feature_matrix (pd.DataFrame): Tabular dataset containing datetime
            features, window features (average lag features), and lag features.

        Returns:
            np.ndarray: Naive forecast
        """
        return feature_matrix["lag_1"].values


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


def split_data(
    data: pd.DataFrame, train_size: float = 0.8, target: str = "target"
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Splits 'data' into train and test sets

    Args:
        data (pd.DataFrame): Tabular, ML-ready dataset
        train_size (float): Percentage of data used for training. Defaults to 0.8.
        target (str, optional): Column name of the target variable. Defaults to "target".

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train set and test set
        feature matrices and target vectors
    """
    try:
        split_idx: int = data.sample(frac=train_size).shape[0]
        split_ts: pd.Timestamp = data.loc[split_idx, "pickup_datetime"]
        features: list[str] = ["hour", "day_of_week"] + [
            col for col in data.columns if "lag" in col
        ]
        x_train: pd.DataFrame = data.query(f"pickup_datetime < '{str(split_ts)}'")[features]
        y_train: pd.Series = data.query(f"pickup_datetime < '{str(split_ts)}'")[target]
        x_test: pd.DataFrame = data.query(f"pickup_datetime >= '{str(split_ts)}'")[features]
        y_test: pd.Series = data.query(f"pickup_datetime >= '{str(split_ts)}'")[target]
        return x_train, y_train, x_test, y_test
    except Exception as e:
        raise e

def train_model(
    data: pd.DataFrame, 
    non_features: list[str] = ["location_id", "pickup_datetime"], 
    target: str = "target", 
    n_folds: int = 5
) -> NaiveForecast | XGBRegressor | LGBMRegressor:
    """Trains and evaluates select ML models and returns the one that produces 
    the lowest average RMSE across 'n_folds'

    Args:
        data (pd.DataFrame): Tabular, ML-ready dataset
        non_features (list[str], optional): List of columns to exclude from training. 
        Defaults to ["location_id", "pickup_datetime"].
        target (str, optional): Column name of the target variable. Defaults to "target".
        n_folds: (int, optional): Number of folds used to train and evaluate each model.
        Defaults to 5.

    Returns:
        NaiveForecast | XGBRegressor | LGBMRegressor: Model that produces the lowest 
        average RMSE across 'n_folds'
    """
    try:
        # create the feature matrix and target vector
        feature_matrix: pd.DataFrame = data.drop(non_features + [target], axis=1)
        target_vector: pd.Series = data[target]

        # a dictionary of models to train and evaluate
        models: dict[str, NaiveForecast | XGBRegressor | LGBMRegressor] = {
            "Baseline": NaiveForecast(),
            "XGBoost": XGBRegressor(n_jobs=-1, early_stopping_rounds=50),
            "LightGBM": LGBMRegressor(verbosity=-1, n_jobs=-1, early_stopping_rounds=50),
        }

        # an empty dictionary to map each model to its average evaluation metric (RMSE)
        report: dict[str, dict[str, float]] = {}
        
        # iterate over each model and train/evaluate it on n_folds
        for model_name, model in tqdm(models.items()):
            horizon: int = int(feature_matrix.shape[0] / (n_folds + 1))
            train_indices: list[int] = [horizon * i for i in range(1, n_folds + 1)]
            val_indices: list[int] = [idx + horizon for idx in train_indices]
            val_indices[-1] = max(val_indices[-1], feature_matrix.shape[0])
            
            # an empty list to store each fold's evaluation metric (RMSE)
            eval_metrics: list[float] = [] 
            for train_idx, val_idx in zip(train_indices, val_indices):
                x_train: pd.DataFrame = feature_matrix.iloc[:train_idx, :]
                y_train: pd.Series = target_vector.iloc[:train_idx]
                x_val: pd.DataFrame = feature_matrix.iloc[train_idx:val_idx, :]
                y_val: pd.Series = target_vector.iloc[train_idx:val_idx]
                if isinstance(model, NaiveForecast):
                    metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")
                elif isinstance(model, XGBRegressor):
                    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                    metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")
                else:
                    model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
                    metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")
                eval_metrics.append(metric)
            report[model_name] = np.mean(eval_metrics)
        best_model: str = (
            pd.DataFrame.from_dict(report, orient="index", columns=["rmse"])
            .sort_values("rmse")
            .index[0]
        )
        return models.get(best_model)
    except Exception as e:
        raise e


def objective(trial: Trial) -> float:
    """Given a user-defined set of hyperparameters, a model is trained
    on five folds of data, and its average validation RMSE is returned

    Args:
        trial (Trial): object that defines the range of each hyperparameter

    Returns:
        float: Average validation RMSE
    """
    try:
        # load the default parameters
        default_params: dict[str, int | str] = TRAIN_CONFIG.get("LGBMRegressor")

        # create the hyperparameter search space
        # NOTE: the 'num_iterations' and 'min_sum_hessian_in_leaf' hyperparameters are ...
        # similar to XGBoost's 'n_estimators' and 'min_child_weight' hyperparameters, respectively
        hyperparams: dict[str, Trial] = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_iterations": trial.suggest_int("num_iterations", 10, 1000),
            # "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_sum_hessian_in_leaf": trial.suggest_int("min_sum_hessian_in_leaf", 0, 50),
        }

        # instantiate an object of type, 'LGBMRegressor'
        model: LGBMRegressor = LGBMRegressor(**(default_params | hyperparams))

        # read in ./data/ml_dataset.parquet
        data: pd.DataFrame = pd.read_parquet(Config.DATA_DIR / "ml_dataset.parquet")

        # extract the training data only
        # NOTE: the training data will be further partitioned into k folds, ...
        # where each fold contains a 'train' set and 'validation' set
        train_features, train_labels, _, _ = split_data(data)

        # get the 'train' set and 'validation' set indices
        n_folds: int = 5
        horizon: int = int(train_features.shape[0] / (n_folds + 1))
        train_indices: list[int] = [horizon * i for i in range(1, n_folds + 1)]
        val_indices: list[int] = [idx + horizon for idx in train_indices]
        val_indices[-1] = max(val_indices[-1], train_features.shape[0])

        # iterate over each fold and compute its 'validation' set RMSE
        metrics: list[float] = []
        for train_idx, val_idx in zip(train_indices, val_indices):
            # split into 'train' and 'validation' sets
            x_train: pd.DataFrame = train_features.iloc[:train_idx, :]
            y_train: pd.Series = train_labels.iloc[:train_idx]
            x_val: pd.DataFrame = train_features.iloc[train_idx:val_idx, :]
            y_val: pd.Series = train_labels.iloc[train_idx:val_idx]

            # fit the model to the 'train' set and evaluate on the 'validation' set
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

            # compute the 'validation' set's RMSE and add it to the 'metrics' list
            metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")
            metrics.append(metric)
        return round(np.mean(metrics), 4)
    except Exception as e:
        raise e


def optimize_model() -> XGBRegressor | LGBMRegressor:
    try:
        # instantiate an object of type, 'LGBMRegressor'
        default_params: dict[str, int | str] = TRAIN_CONFIG.get("LGBMRegressor")
        model: LGBMRegressor = LGBMRegressor(**default_params)

        # create the study and optimize the model's hyperparameters
        model_name: str = model.__str__().split("(")[0]
        study: Study = optuna.create_study(
            study_name=f"{model_name} hyperparameter tuning", direction="minimize"
        )
        study.optimize(objective, n_trials=5)

        # extract the study's 'best' hyperparameters
        best_hyperparams: dict[str, int | float] = study.best_params
        return model.set_params(**best_hyperparams)
    except Exception as e:
        raise e
