# preprocessing/finalizer.py

from typing import Any, Dict
import pandas as pd
from sklearn.base import BaseEstimator

from automl_engine.planning.experiment import ResolvedConfig
from automl_engine.planning.models import ModelSpec
from automl_engine.preprocessing import build_pipeline


def finalize_model(
    best_model_name: str,
    state: Any,
    models: Dict[str, ModelSpec],
    X: pd.DataFrame,
    y: Any,
    resolved: ResolvedConfig,
) -> BaseEstimator:
    """
    Finalize and fit the best-performing model pipeline.

    This function retrieves the best model configuration, rebuilds the pipeline
    if necessary, fits it on the full dataset, and returns the trained pipeline.

    Parameters
    ----------
    best_model_name : str
        Name of the selected best model.
    state : Any
        AutoML state object containing cached pipelines.
    models : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their configuration metadata.
    X : pd.DataFrame
        Feature dataset.
    y : Any
        Target variable.
    resolved : ResolvedConfig
        AutoML configuration object.

    Returns
    -------
    BaseEstimator
        The fitted sklearn-compatible pipeline.
    """
    best_info = models[best_model_name]

    pipeline = state.get_pipeline(best_model_name)

    if pipeline is None:
        pipeline = build_pipeline(
            best_info,
            resolved,
        )

    assert pipeline is not None

    pipeline.fit(X, y)

    return pipeline
