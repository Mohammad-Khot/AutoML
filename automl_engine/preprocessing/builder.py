# preprocessing/builder.py

import pandas as pd
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator

from automl_engine import AutoMLConfig
from automl_engine.utils import inject_seed

from .scalers import select_scaler_strategy
from .encoders import select_encoder_strategy
from .selectors import get_selector
from .imputer import select_imputer_strategy


def init_model(model_info: Dict[str, Any], seed: Optional[int]) -> BaseEstimator:
    """
    Initialize a model instance from the provided model_info dictionary.

    Parameters
    ----------
    model_info : Dict[str, Any]
        Dictionary containing at least a callable under the key "model"
        that returns an uninitialized estimator.
    seed : Optional[int]
        Random seed to inject into the model if supported.

    Returns
    -------
    BaseEstimator
        Instantiated and seeded sklearn-compatible estimator.
    """
    try:
        factory = model_info["model"]
    except KeyError:
        raise ValueError("model_info missing 'model' key")

    if not callable(factory):
        raise TypeError("model_info['model'] must be callable")

    model: BaseEstimator = factory()
    inject_seed(model, seed)

    return model


def build_preprocessor(
    X: pd.DataFrame,
    config: AutoMLConfig,
    model_info: Optional[Dict[str, Any]] = None,
    force_scaling: bool = False,
) -> ColumnTransformer:
    """
    Construct a ColumnTransformer that applies appropriate preprocessing
    strategies for numeric and categorical features.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature dataframe.
    config : AutoMLConfig
        Configuration object controlling preprocessing behavior.
    model_info : Optional[Dict[str, Any]], default=None
        Model metadata used to determine scaling strategy.
    force_scaling : bool, default=False
        Whether to enforce scaling regardless of model requirements.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.
    """
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    scaler = select_scaler_strategy(model_info, X, config, force_scaling)
    encoder = select_encoder_strategy(X, config)
    num_imputer, cat_imputer = select_imputer_strategy(X, config)

    # ---- Numeric Pipeline ----
    num_steps: list[tuple[str, BaseEstimator]] = []

    if num_imputer is not None:
        num_steps.append(("impute", num_imputer))

    if scaler is not None:
        num_steps.append(("scale", scaler))

    num_pipeline: BaseEstimator | str = (
        Pipeline(num_steps) if num_steps else "passthrough"
    )

    # ---- Categorical Pipeline ----
    cat_steps: list[tuple[str, BaseEstimator]] = []

    if cat_imputer is not None:
        cat_steps.append(("impute", cat_imputer))

    if encoder is not None:
        cat_steps.append(("encode", encoder))

    cat_pipeline: BaseEstimator | str = (
        Pipeline(cat_steps) if cat_steps else "passthrough"
    )

    return ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )


def build_pipeline(
    model_info: Dict[str, Any],
    X: pd.DataFrame,
    config: AutoMLConfig,
    seed: Optional[int] = None,
) -> Pipeline:
    """
    Build a complete sklearn Pipeline including preprocessing,
    optional feature selection, and the final model.

    Parameters
    ----------
    model_info : Dict[str, Any]
        Dictionary describing the model factory and related metadata.
    X : pd.DataFrame
        Input feature dataframe.
    config : AutoMLConfig
        Configuration object controlling feature selection and preprocessing.
    seed : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Fully constructed sklearn Pipeline ready for training.
    """
    selector: BaseEstimator = get_selector(
        config.task,
        config.feature_selection,
        n_features=X.shape[1],
    )

    preprocessor: ColumnTransformer = build_preprocessor(
        X,
        config,
        model_info=model_info,
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("select", selector),
            ("model", init_model(model_info, seed)),
        ]
    )
