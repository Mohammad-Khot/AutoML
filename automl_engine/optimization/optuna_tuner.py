# optimization/optuna_tuner.py
import optuna
from typing import Any, Callable, cast
from optuna.trial import Trial
from optuna.study import Study

from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import cross_val_score

from automl_engine.planning.models.registry import MODEL_REGISTRY


def objective(
    trial: Trial,
    pipeline: BaseEstimator,
    X: Any,
    y: Any,
    task: str,
    model_name: str,
    cv: Any,
    scoring: str | Callable,
) -> float:
    """
    Optuna objective function used for hyperparameter optimization.

    This function retrieves the model search space from MODEL_REGISTRY,
    samples hyperparameters using the provided Optuna trial, applies them
    to a cloned pipeline, and evaluates the model using cross-validation.

    Returns the mean cross-validation score.
    """

    spec = MODEL_REGISTRY[task][model_name]

    search_space = spec.search_space

    if search_space is None:
        raise ValueError(
            f"No search_space defined for model '{model_name}' under task '{task}'."
        )

    params = search_space(trial)

    model: BaseEstimator = cast(BaseEstimator, clone(pipeline))
    model.set_params(**params)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    return float(scores.mean())


def run_optuna(
    pipeline: BaseEstimator,
    X: Any,
    y: Any,
    task: str,
    model_name: str,
    cv: Any,
    scoring: str | Callable,
    direction: str,
    config: Any,
    n_trials: int = 100,
    n_jobs: int = 1,
    seed: int = 42,
) -> Study:
    """
    Execute Optuna hyperparameter optimization for a given pipeline.

    Creates an Optuna study with a TPE sampler and Median pruner,
    runs optimization using the defined objective function, and
    returns the completed study object.
    """
    if not config.log:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

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
