# optimization/optuna_tuner.py

import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from automl_engine.planning.models.registry import MODEL_REGISTRY


def objective(
    trial,
    pipeline,
    X,
    y,
    task,
    model_name,
    cv,
    scoring,
):
    """
    Generic Optuna objective.
    Delegates search space definition to MODEL_REGISTRY.
    """

    model_meta = MODEL_REGISTRY[task][model_name]

    search_space = model_meta.get("search_space")

    if search_space is None:
        raise ValueError(
            f"No search_space defined for model '{model_name}' under task '{task}'."
        )

    params = search_space(trial)

    model = clone(pipeline)
    model.set_params(**params)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    return scores.mean()


def run_optuna(
    pipeline,
    X,
    y,
    task,
    model_name,
    cv,
    scoring,
    direction,
    config,
    n_trials=100,
    n_jobs=1,
    seed=42,
):
    """
    Run Optuna hyperparameter optimization.
    """

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study_name = f"{task}_{model_name}_{scoring}_seed{config.seed}"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            pipeline,
            X,
            y,
            task,
            model_name,
            cv,
            scoring,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    return study
