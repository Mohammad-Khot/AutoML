# evaluation/nested.py

import numpy as np

from automl_engine.preprocessing import build_pipeline, build_base_pipeline
from automl_engine.data import infer_task
from automl_engine.core import select_best_model, MODEL_PRIORITY
from automl_engine.evaluation import get_scorer_safe
from .cv import get_cv_object


def nested_cv(X, y, models, outer_cv, config):

    if getattr(outer_cv, "n_splits", 0) < 2:
        raise ValueError(
            "Outer CV has fewer than 2 splits. "
            "Dataset too small for nested CV"
        )

    outer_scores = []
    selected_models = []

    scorer = get_scorer_safe(config.metric)
    task = config.task or infer_task(y)

    from automl_engine.evaluation import evaluate_models

    for i, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n--- OUTER FOLD {i+1}/{outer_cv.n_splits} ---")
        X_train, X_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train, y_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        base_scaled = build_base_pipeline(X_train, config, force_scaling=True)
        base_raw = build_base_pipeline(X_train, config, force_scaling=False)

        if task == "classification" and y_train.value_counts().min() < 2:
            print("[WARN] Fold has classes with <2 samples. Using dummy model.")

            best_name = "dummy" if "dummy" in models else list(models.keys())[0]
            best_info = models[best_name]

            pipeline = build_pipeline(best_info, X_train, config, seed=config.seed, base_scaled=base_scaled, base_raw=base_raw)
            pipeline.fit(X_train, y_train)

            score = scorer(pipeline, X_test, y_test)
            outer_scores.append(score)
            selected_models.append(best_name)
            continue

        safe_inner_cv = get_cv_object(
            task,
            y_train,
            max(2, config.cv_folds - 1),
            config.seed
        )

        state = evaluate_models(X_train, y_train, models, safe_inner_cv, config, base_scaled=base_scaled, base_raw=base_raw)

        best_name = select_best_model(state.scores, MODEL_PRIORITY)
        selected_models.append(best_name)

        best_info = models[best_name]

        pipeline = build_pipeline(best_info, X_train, config)
        pipeline.fit(X_train, y_train)

        score = scorer(pipeline, X_test, y_test)
        outer_scores.append(score)

    return {
        "mean": float(np.mean(outer_scores)),
        "std": float(np.std(outer_scores)),
        "scores": outer_scores,
        "selected_models": selected_models
    }
