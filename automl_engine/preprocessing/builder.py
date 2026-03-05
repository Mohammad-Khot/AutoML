# preprocessing/builder.py

import pandas as pd
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from automl_engine import AutoMLConfig
from automl_engine.utils import inject_seed
from automl_engine.planning.models.spec import ModelSpec

from .scalers import select_scaler_strategy
from .encoders import select_encoder_strategy
from .selectors import get_selector
from .imputer import select_imputer_strategy
from ..planning.config import FeatureSelection


def init_model(spec: ModelSpec, seed: Optional[int]) -> BaseEstimator:
    """
    Initialize a model instance from the provided ModelSpec and inject a seed
    if the model supports random_state.

    Args:
        spec: Model specification containing the model factory.
        seed: Optional random seed for reproducibility.

    Returns:
        BaseEstimator: Instantiated sklearn-compatible model.
    """
    model: BaseEstimator = spec.factory()
    inject_seed(model, seed)
    return model


def build_preprocessor(
    X: pd.DataFrame,
    config: AutoMLConfig,
    spec: Optional[ModelSpec] = None,
    force_scaling: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies preprocessing to numeric and
    categorical columns separately.

    Numeric pipeline may include imputation and scaling.
    Categorical pipeline may include imputation and encoding.

    Args:
        X: Input feature dataframe.
        config: Global AutoML configuration.
        spec: Optional model specification used to determine preprocessing strategy.
        force_scaling: If True, forces scaling even if normally skipped.

    Returns:
        ColumnTransformer: Configured preprocessing transformer.
    """

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    scaler: Optional[BaseEstimator] = select_scaler_strategy(spec, X, config, force_scaling)
    encoder: Optional[TransformerMixin] = select_encoder_strategy(X, config)
    num_imputer: Optional[BaseEstimator]
    cat_imputer: Optional[BaseEstimator]

    num_imputer, cat_imputer = select_imputer_strategy(X, config)

    num_steps: list[tuple[str, BaseEstimator]] = []

    if num_imputer is not None:
        num_steps.append(("impute", num_imputer))

    if scaler is not None:
        num_steps.append(("scale", scaler))

    num_pipeline: BaseEstimator | str = (
        Pipeline(num_steps) if num_steps else "passthrough"
    )

    cat_steps: list[tuple[str, BaseEstimator | TransformerMixin]] = []

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
    spec: ModelSpec,
    X: pd.DataFrame,
    config: AutoMLConfig,
    seed: Optional[int] = None,
) -> Pipeline:
    """
    Construct the full sklearn pipeline including preprocessing,
    feature selection, and the model.

    Args:
        spec: Model specification defining the estimator factory.
        X: Input dataframe used to determine column types and feature count.
        config: AutoML configuration object.
        seed: Optional random seed for reproducibility.

    Returns:
        Pipeline: Complete sklearn pipeline ready for training.
    """

    if config.task is None:
        raise ValueError("config.task must be resolved before building pipeline")

    selector_method: FeatureSelection = config.feature_selection

    selector: BaseEstimator = get_selector(
        config.task,
        selector_method,
        n_features=X.shape[1],
    )

    preprocessor: ColumnTransformer = build_preprocessor(
        X,
        config,
        spec=spec,
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("select", selector),
            ("model", init_model(spec, seed)),
        ]
    )
