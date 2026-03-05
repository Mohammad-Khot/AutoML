# evaluation/evaluate_models.py
from typing import Any, Dict
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

from automl_engine.reporting import log_model_score
from automl_engine.preprocessing import build_pipeline
from automl_engine.runtime.state import AutoMLState


def evaluate_models(
    X: Any,
    y: Any,
    models: Dict[str, Any],
    cv: Any,
    config: Any,
    resolved: Any,
    stage: str,
) -> AutoMLState:
    """
    Evaluate candidate models using cross-validation.

    This function builds a preprocessing + model pipeline for each candidate
    model and evaluates it using sklearn's cross_val_score. It records the
    mean CV score in the AutoMLState object and logs results through the
    reporting system.

    No hyperparameter search is performed here — only evaluation of the
    provided model configurations.

    Parameters
    ----------
    X : Any
        Feature matrix.
    y : Any
        Target vector.
    models : Dict[str, Any]
        Mapping of model names to model specifications/configurations.
    cv : Any
        Cross-validation splitter object.
    config : Any
        Global AutoML configuration object.
    resolved : Any
        Resolved experiment configuration containing the metric.
    stage : str
        Stage label used for logging (e.g., "baseline", "selection").

    Returns
    -------
    AutoMLState
        Updated state object containing model scores and associated pipelines.
    """

    state: AutoMLState = AutoMLState()

    if hasattr(cv, "n_splits") and getattr(cv, "n_splits") < 2:
        log_model_score(
            "ALL",
            "SKIPPED: insufficient CV folds",
            stage=stage,
            log=config.log,
        )
        return state

    metric: str = resolved.metric

    for name, info in models.items():
        try:
            pipeline: BaseEstimator = build_pipeline(
                info,
                X,
                config,
                seed=config.seed,
            )

            scores: np.ndarray = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=metric,
                n_jobs=config.n_jobs,
            )

            scores = np.array(scores)
            mean_score: float = float(np.mean(scores))
            best_params: Dict[str, Any] | None = None

        except Exception as e:
            log_model_score(
                name,
                f"ERROR ({type(e).__name__}: {e})",
                stage=stage,
                log=config.log,
            )
            continue

        log_model_score(
            name,
            round(mean_score, 4),
            stage=stage,
            log=config.log,
        )

        state.update(
            name,
            mean_score,
            pipeline=pipeline,
            params=best_params,
        )

    return state
