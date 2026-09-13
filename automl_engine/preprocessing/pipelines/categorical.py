from sklearn.pipeline import Pipeline

from ..feature_engineering.raw_categorical import RawCategoricalFE
from automl_engine.planning.models.spec import ModelSpec


def build_categorical_pipeline(cat_imputer, encoder, config, model_spec: ModelSpec):
    steps = [
        # Preserve DataFrame semantics for categorical feature engineering.
        ("fe_raw_cat", RawCategoricalFE(config, model_spec=model_spec)),
    ]

    if cat_imputer is not None:
        steps.append(("impute", cat_imputer))

    if encoder is not None:
        steps.append(("encode", encoder))

    return Pipeline(steps)
