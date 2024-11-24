"""This module sets up the project's configurations."""

from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class Paths:
    PROJECT_DIR: PosixPath = Path(__file__).parent.parent.absolute()
    DATA_DIR: PosixPath = PROJECT_DIR / "data"
    ARTIFACTS_DIR: PosixPath = PROJECT_DIR / "artifacts"
    ENV: PosixPath = PROJECT_DIR / ".env"
    CONFIG: PosixPath = PROJECT_DIR / "config.yaml"
    RAW_DATA_URL: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    TAXI_ZONES_SHAPEFILES_URL: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def load_config(path: PosixPath = Paths.CONFIG) -> DictConfig:
    return OmegaConf.load(path)
