# optimization/optimizer.py
from typing import Any, Tuple, Optional

from .optuna_tuner import run_optuna
from automl_engine.evaluation.metrics import get_scorer_safe
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
    """Optimize a model pipeline using Optuna hyperparameter tuning."""
    n_trials = resolve_trials(
        resolved.optuna.n_trials,
        task,
        model_name,
    )

    study = run_optuna(
        pipeline=pipeline,
        X=X,
        y=y,
        task=task,
        model_name=model_name,
        cv=cv,
        scoring=get_scorer_safe(scoring),
        direction=resolved.optuna.direction,
        resolved=resolved,
        n_trials=n_trials,
        n_jobs=resolved.optuna.n_jobs,
        seed=resolved.optuna.seed if resolved.optuna.seed is not None else resolved.runtime.seed,
    )

    pipeline.set_params(**study.best_params)
    pipeline.fit(X, y)

    return pipeline, study


def resolve_trials(
    config_trials: Optional[int],
    task: str,
    model_name: str,
) -> int:
    if config_trials is not None:
        return config_trials

    meta = MODEL_REGISTRY[task][model_name]
    cost = meta.training_cost
    sensitivity = meta.tuning_complexity

    if sensitivity == "low":
        trials = 20
    elif sensitivity == "medium":
        trials = 60
    else:
        trials = 120

    if cost == COST_HIGH:
        trials = int(trials * 0.6)
    elif cost == COST_LOW:
        trials = int(trials * 1.2)

    return trials
