from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, ColumnTransformer
from imblearn.pipeline import Pipeline  # ✅ IMPORTANT

from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.planning.models import ModelSpec
from automl_engine.planning.sampler import build_sampler
from automl_engine.preprocessing.strategies.scalers import select_scaler_strategy
from automl_engine.preprocessing.strategies.encoders import select_encoder_strategy
from automl_engine.preprocessing.pipelines.categorical import build_categorical_pipeline
from automl_engine.preprocessing.pipelines.global_pipeline import build_global_pipeline
from automl_engine.preprocessing.pipelines.numerical import build_numerical_pipeline
from automl_engine.preprocessing.strategies.imputer import select_imputer_strategy
from automl_engine.utils import inject_seed


def init_model(spec: ModelSpec, seed: Optional[int]) -> BaseEstimator:
    model: BaseEstimator = spec.factory()
    inject_seed(model, seed)
    return model


def build_pipeline(
        spec: ModelSpec,
        resolved: ResolvedConfig,
) -> Pipeline:

    scaler = select_scaler_strategy(spec, resolved)
    encoder = select_encoder_strategy(resolved)
    num_imputer, cat_imputer = select_imputer_strategy(resolved)

    num_pipeline = build_numerical_pipeline(num_imputer, scaler, resolved, model_spec=spec)
    cat_pipeline = build_categorical_pipeline(cat_imputer, encoder, resolved, model_spec=spec)

    num_selector = make_column_selector(dtype_include="number")
    cat_selector = make_column_selector(dtype_exclude="number")

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_selector),
        ("cat", cat_pipeline, cat_selector),
    ])

    global_steps = build_global_pipeline(resolved, model_spec=spec)
    sampler = build_sampler(resolved)

    steps = [
        ("preprocess", preprocessor),
    ]

    # ✅ Sampler AFTER preprocessing
    if sampler is not None:
        steps.append(("sampler", sampler))

    # ✅ PCA / global steps AFTER sampler
    steps.extend(global_steps)

    steps.append(("model", init_model(spec, resolved.runtime.seed)))

    return Pipeline(steps)
