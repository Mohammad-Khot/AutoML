# training/workflow.py
from typing import Any, Dict, Tuple, List

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


def execute_training_workflow(
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Dict[str, Any]],
        outer_cv: Any,
        config: AutoMLConfig,
        resolved: ResolvedConfig,
) -> tuple[Any, AutoMLState, list[float], str] | tuple[
    object | Any, AutoMLState, list[str] | list[float], str, dict[str, Figure] | None]:
    # ---------- Standard CV ----------
    if not config.nested_cv:

        if config.log:
            print_section("Standard Cross Validation")

        state = evaluate_models(
            X,
            y,
            models,
            outer_cv,
            config,
            resolved,
            "OUTER_CV",
        )

        best_model_name = max(state.scores, key=state.scores.get)
        best_pipeline = state.get_pipeline(best_model_name)

        return best_pipeline, state, list(state.scores.values()), best_model_name

    # ---------- Nested ----------
    if config.log:
        print_section("Nested Evaluation")

    outer_result = run_nested_cv(
        X,
        y,
        models,
        outer_cv,
        config,
        resolved,
    )

    outer_scores = outer_result["outer_scores"]
    selected_models = outer_result["selected_models"]

    best_model_name = Counter(selected_models).most_common(1)[0][0]

    if config.log:
        print(f"Selected Model (by frequency): {best_model_name}")

    # ---------- Evaluate All Models on Full Data (Leaderboard State) ----------
    state = evaluate_models(
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

    best_info = models[best_model_name]

    tuning_cv = get_cv_object(
        resolved.task,
        y,
        config.cv_folds,
        config.seed,
    )

    pipeline = build_pipeline(
        best_info,
        X,
        config,
        seed=config.seed,
    )

    search_space = best_info.get("search_space")
    optuna_plots = None

    if config.optuna.enabled and search_space is not None:
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
