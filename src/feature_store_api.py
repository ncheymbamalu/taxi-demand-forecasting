"""A script containing functions that return objects that connect to Hopsworks"""

import os

from typing import Any

import hopsworks

from dotenv import load_dotenv
from hopsworks.project import Project
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from omegaconf import OmegaConf

from src.config import Config, load_config

load_dotenv(Config.HOME_DIR / ".env")

HOPSWORKS_CONFIG: dict[str, dict[str, Any] | str] = OmegaConf.to_container(load_config().hopsworks)


def get_feature_store() -> FeatureStore:
    """Connects to the Hopsworks 'taxi_demand_forecasting' project, and
    returns an object that points to its Feature Store

    Returns:
        FeatureStore: Object that points to the 'taxi_demand_forecasting'
        project's Feature Store
    """
    try:
        project: Project = hopsworks.login(
            project=HOPSWORKS_CONFIG.get("project"), api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        return project.get_feature_store()
    except Exception as e:
        raise e


def get_feature_group() -> FeatureGroup:
    """Connects to the Hopsworks 'taxi_demand_forecasting' project's
    Feature Store, and returns an object that points to its Feature Group
    named, 'univariate_time_series'

    Returns:
        FeatureGroup: Object that points to the 'taxi_demand_forecasting'
        project's 'univariate_time_series' Feature Group
    """
    try:
        feature_store: FeatureStore = get_feature_store()
        return feature_store.get_or_create_feature_group(**HOPSWORKS_CONFIG.get("feature_group"))
    except Exception as e:
        raise e
