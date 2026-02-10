from automl_engine.optimization import evaluate_models
from automl_engine.preprocessing import build_pipeline
from automl_engine.data import infer_task
from automl_engine.core import select_best_model, MODEL_PRIORITY
from automl_engine.evaluation import get_scorer_safe
from .cv import get_cv


def nested_cv(X, y, models, outer_cv, config):
    outer_scores = []
    scorer = get_scorer_safe(config.metric)
    task = config.task or infer_task(y)

    for i, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n--- OUTER FOLD {i+1}/{outer_cv.n_splits} ---")
        X_train, X_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train, y_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        if task == "classification" and y_train.value_counts().min() < 2:
            best_name = list(models.keys())[0]
            best_info = models[best_name]

            pipeline = build_pipeline(best_info, X_train, config)
            pipeline.fit(X_train, y_train)

            score = scorer(pipeline, X_test, y_test)
            outer_scores.append(score)
            continue

        safe_inner_cv = get_cv(
            task,
            y_train,
            max(2, config.cv_folds - 1),
            config.seed
        )

        state = evaluate_models(X_train, y_train, models, safe_inner_cv, config)

        best_name = select_best_model(state.scores, MODEL_PRIORITY)
        best_info = models[best_name]

        pipeline = build_pipeline(best_info, X_train, config)
        pipeline.fit(X_train, y_train)
        score = scorer(pipeline, X_test, y_test)
        outer_scores.append(score)

    if getattr(outer_cv, "n_splits", 0) < 2:
        raise ValueError(
            "Outer CV has fewer than 2 splits. "
            "Dataset too small for nested CV"
        )

    return outer_scores
