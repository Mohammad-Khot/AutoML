# training/workflow.py

from typing import Any, Dict

import pandas as pd
from plotly.graph_objs import Figure

from automl_engine.evaluation import evaluate_models
from automl_engine.orchestration.nested import run_nested_cv
from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.reporting import print_section
from automl_engine import AutoMLConfig
from collections import Counter

from automl_engine.runtime.state import AutoMLState
from automl_engine.preprocessing import build_pipeline
from automl_engine.optimization.optimizer import optimize_model
from automl_engine.evaluation import get_cv_object
from automl_engine.planning.models.spec import ModelSpec


def execute_training_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, ModelSpec],
    outer_cv: Any,
    config: AutoMLConfig,
    resolved: ResolvedConfig,
) -> tuple[Any, AutoMLState, list[float], str] | tuple[Any, AutoMLState, list[float], str, dict[str, Figure] | None]:
    """
    Execute the full training workflow including model evaluation, optional nested
    cross-validation, final model fitting, and optional hyperparameter optimization.

    The workflow performs:
    1. Standard cross-validation if nested CV is disabled.
    2. Nested cross-validation for model selection if enabled.
    3. Final evaluation of all models on the full dataset.
    4. Hyperparameter optimization using Optuna for the selected best model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    models : Dict[str, ModelSpec]
        Dictionary mapping model names to their specifications.
    outer_cv : Any
        Cross-validation strategy used for outer evaluation.
    config : AutoMLConfig
        User configuration for the AutoML workflow.
    resolved : ResolvedConfig
        Resolved experiment configuration including task and metric.

    Returns
    -------
    tuple
        If nested CV is disabled:
            (best_pipeline, state, scores, best_model_name)

        If nested CV is enabled:
            (tuned_pipeline, state, outer_scores, best_model_name, optuna_plots)
    """

    # ---------- Standard CV ----------
    if not config.nested_cv:
        if config.log:
            print_section("Standard Cross Validation")

        state: AutoMLState = evaluate_models(
            X,
            y,
            models,
            outer_cv,
            config,
            resolved,
            "OUTER_CV",
        )

        best_model_name: str = max(state.scores, key=state.scores.get)
        best_pipeline: Any = state.get_pipeline(best_model_name)

        return best_pipeline, state, list(state.scores.values()), best_model_name

    # ---------- Nested ----------
    if config.log:
        print_section("Nested Evaluation")

    outer_result: Dict[str, Any] = run_nested_cv(
        X,
        y,
        models,
        outer_cv,
        config,
        resolved,
    )

    outer_scores: list[float] = outer_result["outer_scores"]
    selected_models: list[str] = outer_result["selected_models"]

    best_model_name: str = Counter(selected_models).most_common(1)[0][0]

    if config.log:
        print(f"Selected Model (by frequency): {best_model_name}")

    # ---------- Evaluate All Models on Full Data ----------
    if config.log:
        print_section("Final Fit")

    state: AutoMLState = evaluate_models(
        X,
        y,
        models,
        outer_cv,
        config,
        resolved,
        "FINAL_FIT",
    )

    # ---------- Hyperparameter Optimization ----------
    if config.log:
        print_section("Hyperparameter Optimization")

    best_info: ModelSpec = models[best_model_name]

    tuning_cv: Any = get_cv_object(
        resolved.task,
        y,
        config.cv_folds,
        config.seed,
    )

    pipeline: Any = build_pipeline(
        best_info,
        X,
        config,
        seed=config.seed,
    )

    search_space: Any = best_info.search_space
    optuna_plots: dict[str, Figure] | None = None

    if config.optuna.enabled and search_space is not None:
        tuned_pipeline: Any
        study: Any

        tuned_pipeline, study = optimize_model(
            pipeline=pipeline,
            X=X,
            y=y,
            task=resolved.task,
            model_name=best_model_name,
            cv=tuning_cv,
            scoring=resolved.metric,
            config=config,
        )

        if config.return_optuna_plots and study is not None:
            import optuna.visualization as vis

            optuna_plots = {
                "history": vis.plot_optimization_history(study),
                "importance": vis.plot_param_importances(study),
                "parallel": vis.plot_parallel_coordinate(study),
            }

    else:
        if config.log and config.optuna.enabled:
            print(f"No hyperparameters to tune for '{best_model_name}'. Skipping optimization.")

        tuned_pipeline = pipeline.fit(X, y)

    return tuned_pipeline, state, outer_scores, best_model_name, optuna_plots
