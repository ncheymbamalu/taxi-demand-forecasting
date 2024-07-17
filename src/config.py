from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class Config:
    URL: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    HOME_DIR: PosixPath = Path(__file__).parent.parent
    DATA_DIR: PosixPath = HOME_DIR / "data"
    ARTIFACTS_DIR: PosixPath = HOME_DIR / "artifacts"
    NOTEBOOKS_DIR: PosixPath = HOME_DIR / "notebooks"


def load_config(path: PosixPath = Config.HOME_DIR / "config.yaml") -> DictConfig:
    return OmegaConf.load(path)
