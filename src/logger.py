import logging

from datetime import datetime
from pathlib import PosixPath

from src.paths import PathConfig


LOG_FILE: str = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_DIR: PosixPath = PathConfig.PROJECT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / LOG_FILE,
    format="[%(asctime)s] - %(pathname)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
