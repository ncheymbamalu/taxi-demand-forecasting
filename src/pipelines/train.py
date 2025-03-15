"""This module provides functionality for executing the training pipeline."""

import polars as pl

from xgboost import XGBRegressor

from src.config import Paths, data_config
from src.logger import logger
from src.utils import (
    build_model,
    compute_metrics,
    generate_one_step_forecast,
    save_model,
    save_model_metadata,
    split_data,
    transform_data,
    tune_model,
)


def main() -> None:
    """Evaluates the current model on the latest data and replaces it if its
    one-step forecast produces an R² that's worse than the naive forecast.
    """
    try:
        # name of the target variable column and the column that contains the datetime objects
        target_col: str = data_config.target_column
        temporal_col: str = data_config.temporal_column

        # load the data and transform it into features and targets
        data: pl.DataFrame = pl.read_parquet(Paths.DATA).pipe(transform_data)

        # split the data into a train and evaluation set
        train_data, eval_data = data.pipe(split_data, test_size=1)

        # left join the one-step forecast with the evaluation set
        eval_data = (
            train_data
            .pipe(generate_one_step_forecast)
            .select(["location_id", temporal_col, "forecast"])
            .join(
                other=eval_data.select(["location_id", temporal_col, "lag_1", target_col]),
                how="left",
                on=["location_id", temporal_col]
            )
        )

        # compute the R² for the one-step forecast and the naive forecast
        r2_model: float = compute_metrics(eval_data[target_col], eval_data["forecast"]).get("r2")
        r2_naive: float = compute_metrics(eval_data[target_col], eval_data["lag_1"]).get("r2")

        # replace the current model, if necessary
        if r2_model > r2_naive:
            logger.info("The current model is fine.")
        else:
            logger.info("The current model is unsatisfactory and will be replaced.")
            model, metric = tune_model(data, build_model(data))
            save_model(model)
            save_model_metadata(model, metric)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
