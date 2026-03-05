# orchestration/nested.py

from automl_engine.preprocessing import build_pipeline

from automl_engine.planning.models import (
    select_best_model
)

from automl_engine.evaluation import get_scorer_safe, get_cv_object

from automl_engine.reporting import (
    print_subsection,
    log_model_score,
)

from automl_engine.planning.models.spec import ModelSpec

from typing import Any, Dict, List


def run_nested_cv(
    X: Any,
    y: Any,
    models: Dict[str, ModelSpec],
    outer_cv: Any,
    config: Any,
    resolved: Any,
) -> dict[str, list[float] | list[str]]:
    """
    Execute nested cross-validation.

    The function performs:
    1. Outer CV for estimating generalization performance.
    2. Inner CV for model selection on each outer training split.

    For each outer fold:
    - Inner CV evaluates all candidate models.
    - The best-performing model is selected.
    - That model is retrained on the outer training data.
    - Performance is measured on the outer test set.

    Special handling is applied when classification folds contain
    extremely small class distributions (fallback to a dummy model).

    Returns
    -------
    dict[str, list[float] | list[str]]
        Dictionary containing:
        - "outer_scores": scores obtained on each outer fold
        - "selected_models": model name selected for each fold
    """

    if getattr(outer_cv, "n_splits", 0) < 2:
        raise ValueError(
            "Outer CV has fewer than 2 splits. Dataset too small for nested CV"
        )

    outer_scores: List[float] = []
    selected_models: List[str] = []

    scorer = get_scorer_safe(resolved.metric)
    task = resolved.task

    from automl_engine.evaluation import evaluate_models

    for i, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y), 1
    ):

        if config.log:
            print_subsection(f"Outer Fold {i}/{outer_cv.n_splits}")

        X_train = X.iloc[outer_train_idx]
        X_test = X.iloc[outer_test_idx]
        y_train = y.iloc[outer_train_idx]
        y_test = y.iloc[outer_test_idx]

        if task == "classification" and y_train.value_counts().min() < 2:

            print("[WARN] Fold has classes with <2 samples. Using dummy model.")

            best_name = "dummy" if "dummy" in models else next(iter(models))
            spec = models[best_name]

            pipeline = build_pipeline(
                spec,
                X_train,
                resolved,
                config,
                seed=config.seed,
            )

            pipeline.fit(X_train, y_train)

            score: float = scorer(pipeline, X_test, y_test)

            outer_scores.append(score)
            selected_models.append(best_name)

            continue

        safe_inner_cv = get_cv_object(
            task,
            y_train,
            max(2, config.cv_folds - 1),
            config.seed,
        )

        state = evaluate_models(
            X_train,
            y_train,
            models,
            safe_inner_cv,
            config,
            resolved,
            f"INNER-{i}",
        )

        if not state.scores:
            raise RuntimeError(
                "All models failed during inner CV evaluation."
            )

        best_name: str = select_best_model(state.scores, models)

        selected_models.append(best_name)

        spec = models[best_name]

        pipeline = build_pipeline(
            spec,
            X_train,
            resolved,
            config,
            seed=config.seed,
        )

        pipeline.fit(X_train, y_train)

        score: float = scorer(pipeline, X_test, y_test)

        log_model_score(
            best_name,
            score,
            stage=f"OUTER-{i}",
            log=config.log,
        )

        outer_scores.append(score)

    return {
        "outer_scores": outer_scores,
        "selected_models": selected_models,
    }
