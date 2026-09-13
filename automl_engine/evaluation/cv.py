# evaluation/cv.py

import warnings
from typing import Union

import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
)

from automl_engine.planning.experiment.resolved import ResolvedConfig


def _bounded_folds(requested: int, maximum: int, context: str) -> int:
    if maximum < 2:
        raise ValueError(f"{context} requires at least 2 usable samples per CV constraint.")

    effective = min(requested, maximum)
    if effective != requested:
        warnings.warn(
            f"Reducing CV folds from {requested} to {effective} due to {context}."
        )
    return effective


def get_cv_object(
    target: pd.Series,
    resolved: ResolvedConfig,
    folds: int | None = None,
) -> Union[
    StratifiedKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
]:
    """Construct a safe cross-validation object for the resolved experiment."""
    task = resolved.problem.task
    cv = resolved.cv
    runtime = resolved.runtime

    n_samples = len(target)
    if n_samples < 2:
        raise ValueError("Cross-validation requires at least 2 samples.")

    cv_strategy = cv.strategy
    n_splits = folds if folds is not None else cv.folds
    n_repeats = cv.repeats
    seed = runtime.seed

    if n_splits < 2:
        raise ValueError("Cross-validation requires at least 2 folds.")

    if task == "classification" and target.nunique(dropna=True) < 2:
        raise ValueError("Classification requires at least 2 classes.")

    if cv_strategy == "auto":
        cv_strategy = "stratified" if task == "classification" else "kfold"

    if cv_strategy == "stratified":
        if task != "classification":
            raise ValueError("Stratified CV only works for classification.")

        min_class_count = int(target.value_counts().min())
        effective_folds = _bounded_folds(
            n_splits,
            min_class_count,
            "the smallest class size",
        )
        return StratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=seed,
        )

    if cv_strategy == "kfold":
        effective_folds = _bounded_folds(n_splits, n_samples, "dataset size")
        return KFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=seed,
        )

    if cv_strategy == "repeated":
        if task == "classification":
            min_class_count = int(target.value_counts().min())
            effective_folds = _bounded_folds(
                n_splits,
                min_class_count,
                "the smallest class size",
            )
            return RepeatedStratifiedKFold(
                n_splits=effective_folds,
                n_repeats=n_repeats,
                random_state=seed,
            )

        effective_folds = _bounded_folds(n_splits, n_samples, "dataset size")
        return RepeatedKFold(
            n_splits=effective_folds,
            n_repeats=n_repeats,
            random_state=seed,
        )

    if cv_strategy == "timeseries":
        # TimeSeriesSplit requires n_splits < n_samples.
        effective_folds = _bounded_folds(
            n_splits,
            n_samples - 1,
            "time-series dataset size",
        )
        return TimeSeriesSplit(n_splits=effective_folds)

    raise ValueError(f"Unknown CV strategy: {cv_strategy}")
