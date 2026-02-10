import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

from automl_engine import AutoMLConfig

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler
}


def select_scaler_strategy(
    model_info: dict | None,
    X: pd.DataFrame,
    config: AutoMLConfig,
    force: bool | None = None
):
    num_cols = X.select_dtypes(include="number").columns

    if len(num_cols) == 0:
        return "passthrough"

    if force is not None:
        use_scaler = force
    else:
        use_scaler = (
            config.scaling_mode == "force" or
            (config.scaling_mode == "auto" and model_info and model_info.get("needs_scaling"))
        )

    if use_scaler:
        scaler_cls = SCALERS[config.scaler_type]
        return scaler_cls()

    return "passthrough"
