from automl_engine.utils import log_model_score
from sklearn.model_selection import cross_val_score
from automl_engine.core import AutoMLState
from automl_engine.preprocessing import build_pipeline


def evaluate_models(X, y, models, cv, config):
    state = AutoMLState()

    if hasattr(cv, "n_splits") and cv.n_splits < 2:
        log_model_score("ALL", "SKIPPED: insufficient CV folds", log=config.log)
        return state

    for name, info in models.items():

        pipeline = build_pipeline(info, X, config)

        try:
            scores = cross_val_score(
                pipeline, X, y, cv=cv, scoring=config.metric, n_jobs=config.n_jobs
            )
            mean_score = scores.mean()

        except Exception as e:
            log_model_score(name, f"SKIPPED {e}")
            continue

        log_model_score(name, round(mean_score, 4), log=config.log)
        state.update(name, mean_score)

    return state
