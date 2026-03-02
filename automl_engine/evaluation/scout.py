# optimization/baseline.py

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from automl_engine.preprocessing import build_pipeline
from automl_engine.utils import log_model_score
from automl_engine.evaluation import get_cv_object, get_scorer_safe


def scout_models(X, y, models, cv, config):
    score_table = {}

    if config.scout_fraction < 1.0:
        X_small, _, y_small, _ = train_test_split(
            X,
            y,
            train_size=config.scout_fraction,
            stratify=y if config.task == "classification" else None,
            random_state=config.seed
        )

        X, y = X_small, y_small
        k = len(X)

        cv = get_cv_object(
            config.task,
            y,
            folds=min(config.scout_folds, cv.n_splits),
            seed=config.seed
        )

        log_model_score(
            "SYSTEM",
            f"Using {k} samples and {cv.n_splits} folds",
            stage="SCOUT",
            log=config.log,
        )

    scorer = get_scorer_safe(config.metric)

    for name, info in models.items():
        pipeline = build_pipeline(info, X, config, seed=config.seed)

        try:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scorer,
                n_jobs=config.n_jobs
            )
            mean_score = float(np.mean(scores))

            if np.isfinite(mean_score):
                score_table[name] = mean_score
                log_model_score(name, round(mean_score, 4), stage="SCOUT", log=config.log)

        except Exception as e:
            log_model_score(
                name,
                f"DROPPED ({type(e).__name__})",
                stage="SCOUT",
                log=config.log,
            )

    if not score_table:
        raise ValueError("All models failed during scout phase")

    ranked = sorted(score_table.items(), key=lambda x: x[1], reverse=True)

    top_k = min(getattr(config, "top_k", config.top_k_models), len(ranked))

    selected_names = [name for name, _ in ranked[:top_k]]

    if "dummy" in models and "dummy" not in selected_names:
        selected_names.append("dummy")

    top_models = {name: models[name] for name in selected_names}

    return top_models, score_table
