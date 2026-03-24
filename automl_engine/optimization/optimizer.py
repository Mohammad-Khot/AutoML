# optimization/optimizer.py
from typing import Any, Tuple, Optional

from .optuna_tuner import run_optuna
from automl_engine.planning.models.registry import (
    MODEL_REGISTRY,
    COST_MEDIUM,
    COST_LOW,
    COST_HIGH,
)
from ..planning.experiment import ResolvedConfig


def optimize_model(
    pipeline: Any,
    X: Any,
    y: Any,
    task: str,
    model_name: str,
    cv: Any,
    scoring: str,
    resolved: ResolvedConfig,
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
        resolved: AutoML configuration object.

    Returns:
        Tuple[Any, Any]: The fitted optimized pipeline and the Optuna study.
    """
    n_trials: int = resolve_trials(
        resolved.optuna.n_trials,
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
        direction=resolved.optuna.direction,
        resolved=resolved,
        n_trials=n_trials,
        n_jobs=resolved.optuna.n_jobs,
        seed=resolved.optuna.seed,
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

    meta = MODEL_REGISTRY[task][model_name]

    cost = meta.get("training_cost", COST_MEDIUM)
    sensitivity = meta.get("tuning_complexity", "medium")

    # base from sensitivity
    if sensitivity == "low":
        trials = 20
    elif sensitivity == "medium":
        trials = 60
    else:
        trials = 120

    # adjust for cost
    if cost == COST_HIGH:
        trials = int(trials * 0.6)
    elif cost == COST_LOW:
        trials = int(trials * 1.2)

    return trials
