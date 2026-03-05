# evaluation/evaluate_models.py

from typing import Any, Dict
import numpy as np
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning

from automl_engine.reporting import log_model_score
from automl_engine.preprocessing import build_pipeline
from automl_engine.runtime.state import AutoMLState
from automl_engine.planning.models.spec import ModelSpec


def evaluate_models(
    X: Any,
    y: Any,
    models: Dict[str, ModelSpec],
    cv: Any,
    config: Any,
    resolved: Any,
    stage: str,
) -> AutoMLState:
    """
    Evaluate candidate models using cross-validation.

    Builds preprocessing + model pipelines for each candidate model and
    evaluates them using sklearn's cross_val_score.
    """

    state: AutoMLState = AutoMLState()

    # Guard against degenerate CV
    if hasattr(cv, "n_splits") and getattr(cv, "n_splits") < 2:
        log_model_score(
            "ALL",
            "SKIPPED: insufficient CV folds",
            stage=stage,
            log=config.log,
        )
        return state

    metric: str = resolved.metric

    for name, spec in models.items():

        try:
            pipeline = build_pipeline(
                spec=spec,
                X=X,
                resolved=resolved,
                config=config,
                seed=config.seed,
            )

            # Treat convergence warnings as failures
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)

                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=metric,
                    n_jobs=config.n_jobs,
                )

            scores = np.asarray(scores)
            mean_score: float = float(np.mean(scores))
            best_params = None

        except ConvergenceWarning:
            log_model_score(
                name,
                "SKIPPED: convergence failure",
                stage=stage,
                log=config.log,
            )
            continue

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
