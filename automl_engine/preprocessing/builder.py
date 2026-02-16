# preprocessing/builder.py

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone, BaseEstimator

from automl_engine import AutoMLConfig
from automl_engine.utils import inject_seed


from .scalers import select_scaler_strategy
from .encoders import select_encoder_strategy
from .selectors import get_selector
from .imputer import select_imputer_strategy


# ---------------------------------------------------
# Model Initialization
# ---------------------------------------------------


def init_model(model_info: dict, seed: int | None) -> BaseEstimator:
    try:
        factory = model_info["model"]
    except KeyError:
        raise ValueError("model_info missing 'model' key")

    if not callable(factory):
        raise TypeError("model_info['model'] must be callable")

    model = factory()
    inject_seed(model, seed)

    return model


# ---------------------------------------------------
# Preprocessor Builder (Single Source of Truth)
# ---------------------------------------------------

def build_preprocessor(
        X: pd.DataFrame,
        config: AutoMLConfig,
        model_info: dict | None = None,
        force_scaling: bool = False
) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    scaler = select_scaler_strategy(model_info, X, config, force_scaling)

    encoder = select_encoder_strategy(X, config)
    num_imputer, cat_imputer = select_imputer_strategy(X, config)

    # ---- Numeric Pipeline ----
    num_steps = []

    if num_imputer is not None:
        num_steps.append(("impute", num_imputer))

    if scaler is not None:
        num_steps.append(("scale", scaler))

    num_pipeline = Pipeline(num_steps) if num_steps else "passthrough"

    # ---- Categorical Pipeline ----
    cat_steps = []

    if cat_imputer is not None:
        cat_steps.append(("impute", cat_imputer))

    if encoder is not None:
        cat_steps.append(("encode", encoder))

    cat_pipeline = Pipeline(cat_steps) if cat_steps else "passthrough"

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])


# ---------------------------------------------------
# Full Pipeline Builder
# ---------------------------------------------------

def build_pipeline(
        model_info: dict,
        X: pd.DataFrame,
        config: AutoMLConfig,
        seed: int | None = None,
) -> object | Pipeline:
    selector = get_selector(
        config.task,
        config.feature_selection,
        n_features=X.shape[1]
    )

    preprocessor = build_preprocessor(
        X,
        config,
        model_info=model_info
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("select", selector),
        ("model", init_model(model_info, seed))
    ])
