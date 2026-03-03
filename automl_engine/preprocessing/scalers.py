# preprocessing/scalers.py

from typing import Optional, Union
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)

from automl_engine import AutoMLConfig


SCALERS: dict[str, type[BaseEstimator]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
}


def select_scaler_strategy(
    model_info: Optional[dict],
    X: pd.DataFrame,
    config: AutoMLConfig,
    force: Optional[bool] = None,
) -> Union[BaseEstimator, str]:
    """
    Select and instantiate a scaling strategy based on model requirements
    and configuration settings.

    Parameters
    ----------
    model_info : Optional[dict]
        Dictionary containing metadata about the model. If provided,
        it may include a "needs_scaling" flag used when scaling_mode is "auto".
    X : pd.DataFrame
        Input feature matrix used to detect numeric columns.
    config : AutoMLConfig
        Configuration object containing scaling_mode and scaler_type.
    force : Optional[bool], default=None
        If explicitly set, overrides automatic scaling logic.

    Returns
    -------
    Union[BaseEstimator, str]
        An instantiated scaler if scaling is required, otherwise
        the string "passthrough" when no scaling is applied.
    """
    num_cols = X.select_dtypes(include="number").columns

    if len(num_cols) == 0:
        return "passthrough"

    if force is not None:
        use_scaler = force
    else:
        use_scaler = (
            config.scaling_mode == "force"
            or (
                config.scaling_mode == "auto"
                and model_info is not None
                and model_info.get("needs_scaling")
            )
        )

    if use_scaler:
        scaler_cls = SCALERS[config.scaler_type]
        return scaler_cls()

    return "passthrough"
