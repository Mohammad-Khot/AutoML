# optimization/baseline.py

import numpy as np

from sklearn.model_selection import cross_val_score

from automl_engine.preprocessing import build_pipeline
from automl_engine.core import AutoMLState
from automl_engine.utils import log_model_score
from automl_engine.evaluation import get_cv_object


def filter_by_dummy_once(X, y, models, cv, config, base_scaled, base_raw) -> dict:
    survivors = {}
    mean_score = None

    if getattr(config, "scout_fraction", 1.0) < 1.0:
        n = len(X)
        k = max(50, int(n * config.scout_fraction))
        k = min(k, n)

        rng = np.random.default_rng(config.seed)
        idx = rng.choice(n, size=k, replace=False)

        X = X.iloc[idx]
        y = y.iloc[idx]

        # lightweight CV for scout
        cv = get_cv_object(
            config.task,
            y,
            folds=min(getattr(config, "scout_folds", 3), cv.n_splits),
            seed=config.seed
        )
        print(f"[SCOUT] Using {k} samples and {cv.n_splits} folds")

    dummy_info = models.get("dummy")
    if not dummy_info:
        return models

    dummy_pipe = build_pipeline(dummy_info, X, config, base_scaled=base_scaled, base_raw=base_raw)

    try:
        dummy_scores = cross_val_score(
            dummy_pipe, X, y, cv=cv, scoring=config.metric, n_jobs=config.n_jobs
        )
        dummy_score = float(np.mean(dummy_scores))
        log_model_score("dummy", round(dummy_score, 4), log=config.log)

    except Exception as e:
        print(f"[DUMMY FAILED] {e}")
        return models

    for name, info in models.items():
        if name == "dummy":
            continue

        if info.get("size_sensitive") and X.shape[0] > 30_000:
            print(f"[SIZE DROP] {name} high number of rows")
            continue

        pipeline = build_pipeline(info, X, config, base_scaled=base_scaled, base_raw=base_raw)

        try:
            scores = cross_val_score(
                pipeline, X, y, cv=cv, scoring=config.metric, n_jobs=config.n_jobs
            )
            mean_score = float(np.mean(scores))

        except Exception as e:
            print(f"[GLOBAL DROP] {name} crashed: {e}")

        if not np.isfinite(mean_score):
            print(f"[GLOBAL DROP] {name} non-finite score")
            continue

        log_model_score(name, round(mean_score, 4), log=config.log)

        if mean_score <= dummy_score + config.min_improvement_over_dummy:
            print(f"[GLOBAL DROP] {name} worse than dummy")
        else:
            survivors[name] = info

    survivors["dummy"] = dummy_info
    return survivors

