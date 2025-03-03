"""This module sets up the project's configurations."""

from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class Paths:
    """Configuration for the project's primary directories and filepaths.

    Attributes:
        PROJECT_DIR (PosixPath): Project's root directory.
        LOGS_DIR (PosixPath): Logs directory, ./logs/.
        DATA_DIR (PosixPath): Data directory, ./artifacts/data/.
        MODELS_DIR (PosixPath): Models directory, ./artifacts/models/.
        DATA (PosixPath): Processed data file path, ./artifacts/data/data.parquet.
        MODEL (PosixPath): Trained ML file path, ./artifacts/models/model.pkl.
        ENV (PosixPath): .env file path, ./.env.
        CONFIG (PosixPath): Configuration file path, ./config.yaml.
        TAXI_DATA_API (str): NYC taxi data API.
    """
    PROJECT_DIR: PosixPath = Path(__file__).parent.parent.absolute()
    LOGS_DIR: PosixPath = PROJECT_DIR / "logs"
    DATA_DIR: PosixPath = PROJECT_DIR / "artifacts" / "data"
    MODELS_DIR: PosixPath = PROJECT_DIR / "artifacts" / "models"
    DATA: PosixPath = DATA_DIR / "data.parquet"
    MODEL: PosixPath = MODELS_DIR / "model.pkl"
    ENV: PosixPath = PROJECT_DIR / ".env"
    CONFIG: PosixPath = PROJECT_DIR / "config.yaml"
    TAXI_DATA_API: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def load_config() -> DictConfig:
    """Loads Paths.CONFIG as a DictConfig object.

    Returns:
        DictConfig: Dictionary-like object with user-defined key-values pairs.
    """
    try:
        return OmegaConf.load(Paths.CONFIG)
    except Exception as e:
        raise e


data_config: DictConfig = load_config().data
