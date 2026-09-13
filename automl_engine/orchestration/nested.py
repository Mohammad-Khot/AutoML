# orchestration/nested.py

from typing import Any, List

from automl_engine.planning.experiment import ResolvedConfig
from automl_engine.preprocessing import build_pipeline
from automl_engine.planning.models import select_best_model
from automl_engine.evaluation import get_scorer_safe, get_cv_object, evaluate_models
from automl_engine.reporting import print_subsection, log_model_score


def run_nested_cv(
    X: Any,
    y: Any,
    resolved: ResolvedConfig,
) -> dict[str, list[float] | list[str]]:
    runtime = resolved.runtime
    models = resolved.artifacts.models
    outer_cv = resolved.artifacts.cv_object
    task = resolved.problem.task

    n_outer_splits = outer_cv.get_n_splits(X, y)
    if n_outer_splits < 2:
        raise ValueError("Outer CV has fewer than 2 splits. Dataset too small for nested CV")

    outer_scores: List[float] = []
    selected_models: List[str] = []
    scorer = get_scorer_safe(resolved.problem.metric)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        if runtime.log:
            print_subsection(f"Outer Fold {fold_idx}/{n_outer_splits}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if task == "classification" and y_train.value_counts().min() < 2:
            if runtime.log:
                print("[WARN] Fold has classes with <2 samples. Using fallback model.")

            best_name = "dummy" if "dummy" in models else next(iter(models))
            pipeline = build_pipeline(models[best_name], resolved)
            pipeline.fit(X_train, y_train)

            outer_scores.append(float(scorer(pipeline, X_test, y_test)))
            selected_models.append(best_name)
            continue

        inner_cv = get_cv_object(y_train, resolved)

        state = evaluate_models(
            X_train,
            y_train,
            resolved,
            stage=f"INNER-{fold_idx}",
            cv_override=inner_cv,
        )

        if not state.scores:
            raise RuntimeError("All models failed during inner CV evaluation.")

        best_name = select_best_model(state.scores, models)
        selected_models.append(best_name)

        pipeline = build_pipeline(models[best_name], resolved)
        pipeline.fit(X_train, y_train)

        score = float(scorer(pipeline, X_test, y_test))
        log_model_score(
            best_name,
            score,
            stage=f"OUTER-{fold_idx}",
            log=runtime.log,
        )
        outer_scores.append(score)

    return {
        "outer_scores": outer_scores,
        "selected_models": selected_models,
    }
