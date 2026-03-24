# evaluation/scout.py

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor

from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.preprocessing import build_pipeline
from automl_engine.reporting import log_model_score
from automl_engine.evaluation import get_cv_object, get_scorer_safe


def scout_models(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig,
) -> Tuple[Dict[str, Any], Dict[str, float]]:

    # --- Aliases ---
    task = resolved.problem.task
    metric = resolved.problem.metric
    runtime = resolved.runtime
    search = resolved.search
    models = resolved.artifacts.models
    base_cv = resolved.artifacts.cv_object

    score_table: Dict[str, float] = {}
    cv = base_cv

    # --- Down sampling for scout phase ---
    if search.scout_sample_fraction < 1.0:
        X_sampled, _, y_sampled, _ = train_test_split(
            X,
            y,
            train_size=search.scout_sample_fraction,
            stratify=y if task == "classification" else None,
            random_state=runtime.seed,
        )

        X, y = X_sampled, y_sampled
        n_samples: int = len(X)

        # Rebuild CV for smaller dataset
        cv = get_cv_object(y, resolved)

        log_model_score(
            "SYSTEM",
            f"Using {n_samples} samples and {cv.n_splits} folds",
            stage="SCOUT",
            log=runtime.log,
        )

    scorer = get_scorer_safe(metric)

    # --- Evaluate models ---

    for model_name, model_info in models.items():
        pipeline = build_pipeline(
            model_info,
            resolved,
        )

        try:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scorer,
                n_jobs=runtime.n_jobs,
            )

            mean_score: float = float(np.mean(scores))

            if np.isfinite(mean_score):
                score_table[model_name] = mean_score

                log_model_score(
                    model_name,
                    round(mean_score, 4),
                    stage="SCOUT",
                    log=runtime.log,
                )

        except Exception as e:
            log_model_score(
                model_name,
                f"DROPPED ({type(e).__name__})",
                stage="SCOUT",
                log=runtime.log,
            )

    # --- Safety check ---
    if not score_table:

        if task == "classification":
            model = DummyClassifier(strategy="most_frequent")
        else:
            model = DummyRegressor(strategy="mean")

        model.fit(X, y)

        return (
            {"dummy_fallback": model},
            {}
        )

    # --- Rank models ---
    ranked_models = sorted(
        score_table.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_k: int = min(resolved.models.top_k_models, len(ranked_models))

    selected_names = [name for name, _ in ranked_models[:top_k]]

    if "dummy" in models and "dummy" not in selected_names:
        selected_names.append("dummy")

    top_models: Dict[str, Any] = {
        name: models[name] for name in selected_names
    }

    return top_models, score_table
