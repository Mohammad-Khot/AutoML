from sklearn.model_selection import cross_val_score

from automl_engine.utils import log_model_score
from automl_engine.core import AutoMLState
from automl_engine.preprocessing import build_pipeline
from automl_engine.optimization.param_grid import get_param_grid
from automl_engine.optimization.search_executor import run_grid_search


def evaluate_models(X, y, models, cv, config, resolved):
    """
    Evaluate candidate models using CV.

    Parameters
    ----------
    X, y : dataset
    models : model registry subset
    cv : cross-validator
    config : user runtime policy
    resolved : resolved experiment truth
    """

    state = AutoMLState()

    # ---------- CV sanity ----------
    if hasattr(cv, "n_splits") and cv.n_splits < 2:
        log_model_score(
            "ALL",
            "SKIPPED: insufficient CV folds",
            log=config.log,
        )
        return state

    metric = resolved.metric
    task = resolved.task

    # ---------- Model loop ----------
    for name, info in models.items():

        try:
            # Build preprocessing + estimator
            pipeline = build_pipeline(
                info,
                X,
                config,
                seed=config.seed,
            )

            param_grids = get_param_grid(task)
            param_grid = param_grids.get(name, {})

            # ---------- Hyperparameter search ----------
            if config.search_type == "grid" and param_grid:

                best_pipeline, mean_score, best_params = run_grid_search(
                    pipeline=pipeline,
                    param_grid=param_grid,
                    config=config,
                    X=X,
                    y=y,
                    cv=cv,
                    scoring=metric,
                )

            # ---------- Standard CV ----------
            else:
                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=metric,
                    n_jobs=config.n_jobs,
                )

                mean_score = float(scores.mean())
                best_pipeline = pipeline
                best_params = None

        except Exception as e:
            log_model_score(
                name,
                f"SKIPPED ({type(e).__name__}: {e})",
                log=config.log,
            )
            continue

        # ---------- Logging ----------
        log_model_score(
            name,
            round(mean_score, 4),
            log=config.log,
        )

        state.update(
            name,
            mean_score,
            pipeline=best_pipeline,
            params=best_params,
        )

    return state