# preprocessing/scalers.py

from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)

from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.planning.models import ModelSpec

SCALERS: dict[str, type[BaseEstimator]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
}


def select_scaler_strategy(
    model_info: ModelSpec,
    resolved: ResolvedConfig,
) -> BaseEstimator | None:
    """
    Select and instantiate a scaling strategy using DataInfo.
    """

    data_info = resolved.artifacts.data_info

    # --- no numeric features ---
    if data_info.num_numeric_features == 0:
        return None

    # --- decide whether scaling is needed ---
    mode = resolved.preprocessing.scaling_mode

    if mode == "force":
        use_scaler = True
    elif mode == "auto":
        use_scaler = bool(
            model_info and getattr(model_info, "requires_scaling", False)
        )
    else:
        use_scaler = False

    if not use_scaler:
        return None

    scaler_kind = resolved.preprocessing.scaler_kind

    # --- AUTO SELECTION USING DATAINFO ---
    if scaler_kind == "auto":
        if data_info.is_sparse:
            scaler_kind = "maxabs"

        elif data_info.has_outliers:
            scaler_kind = "robust"

        elif data_info.scale_range_large:
            scaler_kind = "minmax"

        else:
            scaler_kind = "standard"

    if scaler_kind not in SCALERS:
        raise ValueError(f"Unknown scaler_kind: {scaler_kind}")

    return SCALERS[scaler_kind]()
