# evaluation/scout.py

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from automl_engine.preprocessing import build_pipeline
from automl_engine.reporting import log_model_score
from automl_engine.evaluation import get_cv_object, get_scorer_safe


def scout_models(
    X: Any,
    y: Any,
    models: Dict[str, Any],
    cv: Any,
    config: Any
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Perform a lightweight scouting phase to rank candidate models using cross-validation.

    This function optionally sub samples the dataset (based on `config.scout_fraction`)
    to quickly estimate model performance. It evaluates each model using the provided
    cross-validation strategy and scoring metric, ranks them by mean score, and
    returns the top-performing models along with their scores.

    Parameters
    ----------
    X : Any
        Feature matrix.
    y : Any
        Target vector.
    models : Dict[str, Any]
        Dictionary mapping model names to model configuration objects.
    cv : Any
        Cross-validation object with an `n_splits` attribute.
    config : Any
        Configuration object containing scouting parameters such as:
        - scout_fraction
        - scout_folds
        - seed
        - task
        - metric
        - n_jobs
        - log
        - top_k or top_k_models

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, float]]
        A tuple containing:
        - Dictionary of selected top models after scouting.
        - Dictionary mapping model names to their mean cross-validation scores.
    """
    score_table: Dict[str, float] = {}

    if config.scout_fraction < 1.0:
        X_small, _, y_small, _ = train_test_split(
            X,
            y,
            train_size=config.scout_fraction,
            stratify=y if config.task == "classification" else None,
            random_state=config.seed,
        )

        X, y = X_small, y_small
        k: int = len(X)

        cv = get_cv_object(
            config.task,
            y,
            folds=min(config.scout_folds, cv.n_splits),
            seed=config.seed,
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
                n_jobs=config.n_jobs,
            )
            mean_score: float = float(np.mean(scores))

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

    top_k: int = min(getattr(config, "top_k", config.top_k_models), len(ranked))

    selected_names = [name for name, _ in ranked[:top_k]]

    if "dummy" in models and "dummy" not in selected_names:
        selected_names.append("dummy")

    top_models: Dict[str, Any] = {name: models[name] for name in selected_names}

    return top_models, score_table
