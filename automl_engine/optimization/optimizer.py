# optimization/optimizer.py
from typing import Any, Tuple, Optional

from .optuna_tuner import run_optuna
from automl_engine.planning.models.registry import (
    MODEL_REGISTRY,
    COST_MEDIUM,
    COST_LOW,
    COST_HIGH,
)


def optimize_model(
    pipeline: Any,
    X: Any,
    y: Any,
    task: str,
    model_name: str,
    cv: Any,
    scoring: str,
    config: Any,
) -> Tuple[Any, Any]:
    """
    Optimize a model pipeline using Optuna hyperparameter tuning.

    Determines the number of trials based on configuration or model
    compute cost, runs the Optuna study, applies the best parameters
    to the pipeline, and fits the final model on the full dataset.

    Args:
        pipeline: The sklearn pipeline containing preprocessing and model.
        X: Feature dataset.
        y: Target labels.
        task: Task type (e.g., "classification", "regression").
        model_name: Name of the model in the registry.
        cv: Cross-validation strategy.
        scoring: Scoring metric used for evaluation.
        config: AutoML configuration object.

    Returns:
        Tuple[Any, Any]: The fitted optimized pipeline and the Optuna study.
    """
    n_trials: int = resolve_trials(
        config.optuna.n_trials,
        task,
        model_name,
    )

    study: Any = run_optuna(
        pipeline=pipeline,
        X=X,
        y=y,
        task=task,
        model_name=model_name,
        cv=cv,
        scoring=scoring,
        direction=config.optuna.direction,
        config=config,
        n_trials=n_trials,
        n_jobs=config.optuna.n_jobs,
        seed=config.optuna.seed,
    )

    pipeline.set_params(**study.best_params)
    pipeline.fit(X, y)

    return pipeline, study


def resolve_trials(
    config_trials: Optional[int],
    task: str,
    model_name: str,
) -> int:
    """
    Resolve the number of Optuna trials based on configuration or model cost.

    If the user provides a fixed number of trials in the configuration,
    that value is used. Otherwise, the trial count is determined from
    the model registry based on the model's compute cost.

    Args:
        config_trials: Optional user-defined number of trials.
        task: Task type used to access the model registry.
        model_name: Name of the model in the registry.

    Returns:
        int: Number of Optuna trials to run.
    """
    if config_trials is not None:
        return config_trials

    meta: dict = MODEL_REGISTRY[task][model_name]
    cost: str = meta.get("compute_cost", COST_MEDIUM)

    if cost == COST_LOW:
        return 30
    if cost == COST_MEDIUM:
        return 75
    if cost == COST_HIGH:
        return 150

    return 75
