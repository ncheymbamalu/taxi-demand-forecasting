from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class PathConfig:
    PROJECT_DIR: PosixPath = Path(__file__).parent.parent.absolute()
    DATA_DIR: PosixPath = PROJECT_DIR / "data"
    ARTIFACTS_DIR: PosixPath = PROJECT_DIR / "artifacts"
    RAW_DATA_URL: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    TAXI_ZONES_SHAPEFILES_URL: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def load_config(path: PosixPath = PathConfig.PROJECT_DIR / "config.yaml") -> DictConfig:
    return OmegaConf.load(path)
