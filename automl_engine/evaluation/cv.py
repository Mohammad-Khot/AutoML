# evaluation/cv.py

from typing import Literal
import warnings

import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
)


def get_cv_object(
    task: str,
    y: pd.Series,
    folds: int,
    seed: int,
    strategy: Literal["auto", "stratified", "kfold", "repeated", "timeseries"] = "auto",
    repeats: int = 3,
) -> StratifiedKFold | KFold | RepeatedKFold | RepeatedStratifiedKFold | TimeSeriesSplit:
    """
    Create and return an appropriate scikit-learn cross-validation splitter
    based on the task type and chosen strategy.

    Parameters
    ----------
    task : Literal["classification", "regression"]
        The machine learning task type.
    y : pd.Series
        Target variable used for validation checks (e.g., class distribution).
    folds : int
        Number of cross-validation splits.
    seed : int
        Random seed for reproducibility.
    strategy : Literal["auto", "stratified", "kfold", "repeated", "timeseries"], default="auto"
        Cross-validation strategy to use.
    repeats : int, default=3
        Number of repeats for repeated cross-validation strategies.

    Returns
    -------
    StratifiedKFold | KFold | RepeatedKFold | RepeatedStratifiedKFold | TimeSeriesSplit
        An instantiated scikit-learn cross-validation splitter.
    """
    if task == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least 2 classes.")

    if strategy == "auto":
        strategy = "stratified" if task == "classification" else "kfold"

    if strategy == "stratified":
        min_class = y.value_counts().min()
        safe_folds = max(2, min(folds, int(min_class)))

        if safe_folds != folds:
            warnings.warn(
                f"Reducing CV folds from {folds} to {safe_folds} due to small class size."
            )

        return StratifiedKFold(
            n_splits=safe_folds,
            shuffle=True,
            random_state=seed,
        )

    if strategy == "kfold":
        return KFold(
            n_splits=max(2, folds),
            shuffle=True,
            random_state=seed,
        )

    if strategy == "repeated":
        if task == "classification":
            return RepeatedStratifiedKFold(
                n_splits=folds,
                n_repeats=repeats,
                random_state=seed,
            )
        return RepeatedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=seed,
        )

    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=max(2, folds))

    raise ValueError(f"Unknown CV strategy: {strategy}")
