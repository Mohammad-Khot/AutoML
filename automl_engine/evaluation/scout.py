# evaluation/scout.py

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.preprocessing import build_pipeline
from automl_engine.reporting import log_model_score
from automl_engine.evaluation.cv import get_cv_object
from automl_engine.evaluation.metrics import get_scorer_safe


def scout_models(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    task = resolved.problem.task
    runtime = resolved.runtime
    search = resolved.search
    models = resolved.artifacts.models

    score_table: Dict[str, float] = {}

    # The scout stage has its own fold budget. It must not silently reuse the
    # full experiment fold count.
    cv = get_cv_object(y, resolved, folds=search.scout_folds)

    if search.scout_sample_fraction < 1.0:
        try:
            X_sampled, _, y_sampled, _ = train_test_split(
                X,
                y,
                train_size=search.scout_sample_fraction,
                stratify=y if task == "classification" else None,
                random_state=runtime.seed,
            )

            # Only accept the reduced sample when a valid CV object can be
            # constructed. Tiny/imbalanced datasets fall back to full data.
            sampled_cv = get_cv_object(
                y_sampled,
                resolved,
                folds=search.scout_folds,
            )
            X, y, cv = X_sampled, y_sampled, sampled_cv

            log_model_score(
                "SYSTEM",
                f"Using {len(X)} samples for scout evaluation",
                stage="SCOUT",
                log=runtime.log,
            )
        except ValueError:
            cv = get_cv_object(y, resolved, folds=search.scout_folds)

    scorer = get_scorer_safe(resolved.problem.metric)

    for model_name, model_info in models.items():
        pipeline = build_pipeline(model_info, resolved)
        try:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scorer,
                n_jobs=runtime.n_jobs,
            )
            mean_score = float(np.mean(scores))

            if np.isfinite(mean_score):
                score_table[model_name] = mean_score
                log_model_score(
                    model_name,
                    round(mean_score, 4),
                    stage="SCOUT",
                    log=runtime.log,
                )

        except Exception as exc:
            log_model_score(
                model_name,
                f"DROPPED ({type(exc).__name__})",
                stage="SCOUT",
                log=runtime.log,
            )

    if not score_table:
        # Keep the return type consistent: callers expect ModelSpec values,
        # never fitted estimator instances.
        if "dummy" in models:
            return {"dummy": models["dummy"]}, {}
        first_name = next(iter(models))
        return {first_name: models[first_name]}, {}

    ranked_models = sorted(
        score_table.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_k = min(resolved.models.top_k_models, len(ranked_models))
    selected_names = [name for name, _ in ranked_models[:top_k]]

    if "dummy" in models and "dummy" not in selected_names:
        selected_names.append("dummy")

    top_models = {name: models[name] for name in selected_names}
    return top_models, score_table
