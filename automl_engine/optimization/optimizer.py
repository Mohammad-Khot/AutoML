from .optuna_tuner import run_optuna
from automl_engine.planning.models.registry import (
    MODEL_REGISTRY,
    COST_MEDIUM,
    COST_LOW,
    COST_HIGH,
)


def optimize_model(
    pipeline,
    X,
    y,
    task,
    model_name,
    cv,
    scoring,
    config,
):
    n_trials = resolve_trials(
        config.optuna.n_trials,
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


def resolve_trials(config_trials, task, model_name):
    if config_trials is not None:
        return config_trials

    meta = MODEL_REGISTRY[task][model_name]
    cost = meta.get("compute_cost", COST_MEDIUM)

    if cost == COST_LOW:
        return 30
    if cost == COST_MEDIUM:
        return 75
    if cost == COST_HIGH:
        return 150

    return 75
