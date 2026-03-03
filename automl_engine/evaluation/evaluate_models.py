# evaluation/evaluate_models.py

from typing import Any, Dict
import numpy as np
from sklearn.model_selection import cross_val_score

from automl_engine.reporting import log_model_score
from automl_engine.preprocessing import build_pipeline
from automl_engine.optimization.param_grid import get_param_grid
from automl_engine.optimization.search_executor import run_grid_search
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
    Evaluate candidate models using cross-validation and optional hyperparameter search.

    Parameters
    ----------
    X : Any
        Feature matrix.
    y : Any
        Target vector.
    models : Dict[str, Any]
        Dictionary mapping model names to model metadata/definitions.
    cv : Any
        Cross-validation splitter instance.
    config : Any
        Runtime configuration object containing search strategy, logging, and parallelism settings.
    resolved : Any
        Resolved experiment configuration containing task type and evaluation metric.
    stage : str
        Current stage of execution for logging purposes.

    Returns
    -------
    AutoMLState
        Updated state object containing evaluated models, scores, pipelines, and parameters.
    """
    state: AutoMLState = AutoMLState()

    # ---------- CV sanity ----------
    if hasattr(cv, "n_splits") and getattr(cv, "n_splits") < 2:
        log_model_score(
            "ALL",
            "SKIPPED: insufficient CV folds",
            stage=stage,
            log=config.log,
        )
        return state

    metric: str = resolved.metric
    task: str = resolved.task

    # ---------- Model loop ----------
    for name, info in models.items():
        try:
            pipeline: Any = build_pipeline(
                info,
                X,
                config,
                seed=config.seed,
            )

            param_grids: Dict[str, Dict[str, Any]] = get_param_grid(task)
            param_grid: Dict[str, Any] = param_grids.get(name, {})

            # ---------- Hyperparameter search ----------
            if config.search_type == "grid" and param_grid:
                best_pipeline, mean_score, best_params = run_grid_search(
                    pipeline=pipeline,
                    param_grid=param_grid,
                    config=config,
                    X=X,
                    y=y,
                    cv=cv,
                )
            # ---------- Standard CV ----------
            else:
                scores: np.ndarray = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=metric,
                    n_jobs=config.n_jobs,
                )

                mean_score: float = float(np.mean(scores))
                best_pipeline = pipeline
                best_params: Dict[str, Any] | None = None

        except Exception as e:
            log_model_score(
                name,
                f"ERROR ({type(e).__name__}: {e})",
                stage=stage,
                log=config.log,
            )
            continue

        # ---------- Logging ----------
        log_model_score(
            name,
            round(mean_score, 4),
            stage=stage,
            log=config.log,
        )

        state.update(
            name,
            mean_score,
            pipeline=best_pipeline,
            params=best_params,
        )

    return state