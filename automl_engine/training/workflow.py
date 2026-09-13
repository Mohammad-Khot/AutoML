# training/workflow.py

from typing import Any, Dict

import pandas as pd
from plotly.graph_objs import Figure
from collections import Counter

from sklearn.base import BaseEstimator

from automl_engine.evaluation import evaluate_models, get_cv_object
from automl_engine.orchestration.nested import run_nested_cv
from automl_engine.planning.experiment import ResolvedConfig
from automl_engine.planning.models import select_best_model
from automl_engine.reporting import print_section

from automl_engine.runtime.state import AutoMLState
from automl_engine.preprocessing import build_pipeline
from automl_engine.optimization.optimizer import optimize_model


def execute_training_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig,
) -> tuple[Any, AutoMLState, list[float], str, dict[str, Figure] | None]:
    runtime = resolved.runtime
    cv_config = resolved.cv
    models = resolved.artifacts.models
    task = resolved.problem.task
    metric = resolved.problem.metric

    optuna_plots: dict[str, Figure] | None = None

    if not cv_config.use_nested_cv:
        if runtime.log:
            print_section("Standard Cross Validation")

        state = evaluate_models(X, y, resolved, "OUTER_CV")

        if not state.scores:
            raise RuntimeError("No models were successfully evaluated.")

        best_model_name = select_best_model(state.scores, models)
        best_pipeline: BaseEstimator = state.get_pipeline(best_model_name)

        # cross_val_score fits clones of the pipeline, not the pipeline stored
        # in AutoMLState. Fit the selected pipeline on all data before exposing
        # it through AutoMLEngine.predict().
        best_pipeline.fit(X, y)

        return (
            best_pipeline,
            state,
            [],
            best_model_name,
            None,
        )

    if runtime.log:
        print_section("Nested Evaluation")

    outer_result: Dict[str, Any] = run_nested_cv(X, y, resolved)
    outer_scores = outer_result["outer_scores"]
    selected_models = outer_result["selected_models"]

    if not selected_models:
        raise RuntimeError("Nested CV did not select any model.")

    best_model_name = Counter(selected_models).most_common(1)[0][0]

    if runtime.log:
        print(f"Selected Model (by frequency): {best_model_name}")
        print_section("Final Fit")

    state = evaluate_models(X, y, resolved, "FINAL FIT")
    if not state.scores:
        raise RuntimeError("No models were successfully evaluated during final fit.")

    best_info = models[best_model_name]
    pipeline: Any = build_pipeline(best_info, resolved)
    hyperparameter_space = best_info.hyperparameter_space

    if resolved.optuna.enabled and hyperparameter_space is not None:
        if runtime.log:
            print_section("Hyperparameter Optimization")

        tuning_cv = get_cv_object(y, resolved)
        tuned_pipeline, study = optimize_model(
            pipeline=pipeline,
            X=X,
            y=y,
            task=task,
            model_name=best_model_name,
            cv=tuning_cv,
            scoring=metric,
            resolved=resolved,
        )

        if study is not None and resolved.generate_optuna_plots:
            import optuna.visualization as vis

            optuna_plots = {
                "history": vis.plot_optimization_history(study),
                "importance": vis.plot_param_importances(study),
                "parallel": vis.plot_parallel_coordinate(study),
            }
    else:
        tuned_pipeline = pipeline.fit(X, y)

    return (
        tuned_pipeline,
        state,
        outer_scores,
        best_model_name,
        optuna_plots,
    )
