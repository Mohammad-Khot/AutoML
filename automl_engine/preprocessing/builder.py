# preprocessing/builder.py

import pandas as pd
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.utils import inject_seed
from automl_engine.planning.models.spec import ModelSpec

from .scalers import select_scaler_strategy
from .encoders import select_encoder_strategy
from .selectors import get_selector
from .imputer import select_imputer_strategy
from ..planning.config import FeatureSelection
from automl_engine import AutoMLConfig


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

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    scaler = select_scaler_strategy(spec, X, config, force_scaling)
    encoder = select_encoder_strategy(X, config)

    num_imputer, cat_imputer = select_imputer_strategy(X, config)

    num_steps = []

    if num_imputer is not None:
        num_steps.append(("impute", num_imputer))

    if scaler is not None:
        num_steps.append(("scale", scaler))

    num_pipeline = Pipeline(num_steps) if num_steps else "passthrough"

    cat_steps = []

    if cat_imputer is not None:
        cat_steps.append(("impute", cat_imputer))

    if encoder is not None:
        cat_steps.append(("encode", encoder))

    cat_pipeline = Pipeline(cat_steps) if cat_steps else "passthrough"

    transformers = []

    if len(num_cols) > 0:
        transformers.append(("num", num_pipeline, num_cols))

    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipeline, cat_cols))

    if not transformers:
        raise RuntimeError("No usable feature columns detected.")

    return ColumnTransformer(transformers)


def build_pipeline(
    spec: ModelSpec,
    X: pd.DataFrame,
    resolved: ResolvedConfig,
    config: AutoMLConfig,
    seed: Optional[int] = None,
) -> Pipeline:

    selector_method: FeatureSelection = config.feature_selection

    selector = get_selector(
        resolved.task,
        selector_method,
        n_features=X.shape[1],
    )

    preprocessor = build_preprocessor(
        X,
        config,
        spec=spec,
    )

    steps = [
        ("preprocess", preprocessor),
    ]

    if selector is not None:
        steps.append(("select", selector))

    steps.append(("model", init_model(spec, seed)))

    return Pipeline(steps)