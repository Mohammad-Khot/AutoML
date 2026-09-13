from sklearn.pipeline import Pipeline

from ..feature_engineering.raw_numeric import RawNumericFE
from automl_engine.planning.models.spec import ModelSpec


def build_numerical_pipeline(num_imputer, scaler, resolved, model_spec: ModelSpec):
    steps = [
        # Raw feature engineering needs column names and therefore must run
        # while the input is still a DataFrame. sklearn imputers emit ndarrays.
        ("fe_raw_num", RawNumericFE(resolved, model_spec=model_spec)),
    ]

    if num_imputer is not None:
        steps.append(("impute", num_imputer))

    if scaler is not None:
        steps.append(("scale", scaler))

    return Pipeline(steps)
