# training/workflow.py

from typing import Any, Dict

import pandas as pd
from plotly.graph_objs import Figure
from collections import Counter

from sklearn.base import BaseEstimator

from automl_engine.evaluation import evaluate_models, get_cv_object
from automl_engine.orchestration.nested import run_nested_cv
from automl_engine.planning.experiment import ResolvedConfig
from automl_engine.reporting import print_section

from automl_engine.runtime.state import AutoMLState
from automl_engine.preprocessing import build_pipeline
from automl_engine.optimization.optimizer import optimize_model


def execute_training_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig,
) -> tuple[Any, AutoMLState, list[float], str, dict[str, Figure] | None]:

    # --- Aliases ---
    runtime = resolved.runtime
    cv_config = resolved.cv
    models = resolved.artifacts.models
    task = resolved.problem.task
    metric = resolved.problem.metric

    optuna_plots: dict[str, Figure] | None = None

    # ---------- Standard CV ----------
    if not cv_config.use_nested_cv:
        if runtime.log:
            print_section("Standard Cross Validation")

        state: AutoMLState = evaluate_models(
            X,
            y,
            resolved,
            "OUTER_CV",
        )

        if not state.scores:
            raise RuntimeError("No models were successfully evaluated.")

        best_model_name: str = max(state.scores, key=state.scores.get)
        best_pipeline: BaseEstimator = state.get_pipeline(best_model_name)

        return (
            best_pipeline,
            state,
            list(state.scores.values()),
            best_model_name,
            None,
        )

    # ---------- Nested ----------
    if runtime.log:
        print_section("Nested Evaluation")

    outer_result: Dict[str, Any] = run_nested_cv(X, y, resolved)

    outer_scores: list[float] = outer_result["outer_scores"]
    selected_models: list[str] = outer_result["selected_models"]

    best_model_name: str = Counter(selected_models).most_common(1)[0][0]

    if runtime.log:
        print(f"Selected Model (by frequency): {best_model_name}")

    # ---------- Evaluate All Models on Full Data ----------
    if runtime.log:
        print_section("Final Fit")

    state: AutoMLState = evaluate_models(X, y, resolved, "FINAL FIT")

    # ---------- Hyperparameter Optimization ----------
    if runtime.log:
        print_section("Hyperparameter Optimization")

    best_info = models[best_model_name]

    tuning_cv = get_cv_object(y, resolved)

    pipeline: Any = build_pipeline(
        best_info,
        resolved,
    )

    hyperparameter_space = best_info.hyperparameter_space

    if hyperparameter_space is not None:
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

        if study is not None:
            import optuna.visualization as vis

            optuna_plots = {
                "history": vis.plot_optimization_history(study),
                "importance": vis.plot_param_importances(study),
                "parallel": vis.plot_parallel_coordinate(study),
            }

    else:
        if runtime.log:
            print(
                f"No hyperparameters to tune for '{best_model_name}'. Skipping optimization."
            )

        tuned_pipeline = pipeline.fit(X, y)

    return (
        tuned_pipeline,
        state,
        outer_scores,
        best_model_name,
        optuna_plots,
    )
