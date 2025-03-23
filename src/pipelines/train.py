"""This module provides functionality for executing the training pipeline."""

import polars as pl

from src.config import Paths
from src.logger import logger
from src.utils import (
    build_model,
    evaluate_model,
    save_model,
    save_model_metadata,
    transform_data,
    tune_model,
)


def main() -> None:
    """Evaluates the current model on the latest data and replaces it if its
    performance is worse than the naive forecast.
    """
    try:
        if evaluate_model():
            logger.info("The current model is fine.")
        else:
            data: pl.DataFrame = pl.read_parquet(Paths.DATA).pipe(transform_data)
            logger.info("The current model is unsatisfactory and will be replaced.")
            model, metric = tune_model(data, build_model(data))
            save_model(model)
            save_model_metadata(model, metric)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
