from automl_engine.utils import log_model_score
from sklearn.model_selection import cross_val_score
from automl_engine.core import AutoMLState
from automl_engine.preprocessing import build_pipeline
from automl_engine.optimization.param_grid import get_param_grid
from automl_engine.optimization.search_executor import run_grid_search


def evaluate_models(X, y, models, cv, config):
    state = AutoMLState()

    if hasattr(cv, "n_splits") and cv.n_splits < 2:
        log_model_score("ALL", "SKIPPED: insufficient CV folds", log=config.log)
        return state

    for name, info in models.items():

        pipeline = build_pipeline(info, X, config)

        try:
            param_grids = get_param_grid(config.task)
            param_grid = param_grids.get(name, {})

            try:
                if config.search_type == "grid" and param_grid:

                    best_pipeline, mean_score, best_params = run_grid_search(
                        pipeline,
                        param_grid,
                        config,
                        X,
                        y,
                        cv
                    )

                else:

                    scores = cross_val_score(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring=config.metric,
                        n_jobs=config.n_jobs
                    )
                    mean_score = scores.mean()
                    best_pipeline = pipeline
                    best_params = None

            except Exception as e:
                log_model_score(name, f"SKIPPED {e}")
                continue


        except Exception as e:
            log_model_score(name, f"SKIPPED {e}")
            continue

        log_model_score(name, round(mean_score, 4), log=config.log)
        state.update(name, mean_score, pipeline=best_pipeline, params=best_params)

    return state
