# optimization/optuna_tuner.py
import inspect
import optuna
from typing import Any, Callable, cast, Optional
from optuna.trial import Trial
from optuna.study import Study

from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import cross_val_score

from automl_engine.planning.experiment import ResolvedConfig
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
    spec = MODEL_REGISTRY[task][model_name]
    hyperparameter_space = spec.hyperparameter_space

    if hyperparameter_space is None:
        raise ValueError(
            f"No hyperparameter_space defined for model '{model_name}' under task '{task}'."
        )

    # Some no-op spaces in the registry are intentionally zero-argument
    # callables. Support both those and normal Optuna trial factories.
    if len(inspect.signature(hyperparameter_space).parameters) == 0:
        params = hyperparameter_space()
    else:
        params = hyperparameter_space(trial)

    model = cast(BaseEstimator, clone(pipeline))
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
    resolved: ResolvedConfig,
    n_trials: int = 100,
    n_jobs: int = 1,
    seed: Optional[int] = 42,
) -> Study:
    if not resolved.runtime.log:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()

    study_name = f"{task}_{model_name}_{resolved.problem.metric}_seed{resolved.runtime.seed}"

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
        timeout=resolved.search.time_budget_soft,
    )

    return study
