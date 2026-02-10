# preprocessing/builder.py
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from automl_engine import AutoMLConfig

from .scalers import select_scaler_strategy
from .encoders import select_encoder_strategy
from .selectors import get_selector


def build_base_pipeline(X: pd.DataFrame, config: AutoMLConfig, force_scaling: bool):
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    scaler = select_scaler_strategy(
        model_info=None,
        X=X,
        config=config,
        force=force_scaling
    )

    encoder = select_encoder_strategy(X, config)

    selector = get_selector(
        config.task,
        config.feature_selection,
        n_features=X.shape[1]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, num_cols),
            ("cat", encoder, cat_cols)
        ]
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("select", selector)
    ])


def attach_model(base_pipe: Pipeline, model_info: dict) -> object:
    pipe = clone(base_pipe)
    pipe.steps.append(("model", model_info["model"]()))
    return pipe


def build_pipeline(
        model_info: dict,
        X: pd.DataFrame,
        config: AutoMLConfig,
        base_scaled: Pipeline | None = None,
        base_raw: Pipeline | None = None
) -> object | Pipeline:
    if base_scaled is not None and base_raw is not None:
        base = base_scaled if model_info["needs_scaling"] else base_raw
        return attach_model(base, model_info)

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    scaler = select_scaler_strategy(model_info, X, config)
    encoder = select_encoder_strategy(X, config)

    selector = get_selector(
        config.task,
        config.feature_selection,
        n_features=X.shape[1]
    )

    preprocessor = ColumnTransformer([
        ("num", scaler, num_cols),
        ("cat", encoder, cat_cols)
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        ("select", selector),
        ("model", model_info["model"]())
    ])
