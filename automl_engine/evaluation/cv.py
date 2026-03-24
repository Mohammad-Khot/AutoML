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


def get_cv_object(
    target: pd.Series,
    resolved: ResolvedConfig
) -> Union[
    StratifiedKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
]:
    """
        Construct and return a cross-validation object based on the resolved configuration.

        This function selects the appropriate cross-validation strategy depending on the
        task type (classification or regression) and user-defined configuration. It also
        applies safeguards such as adjusting the number of folds for imbalanced classification
        datasets to prevent runtime errors.

        Supported strategies include:
        - StratifiedKFold (classification only)
        - KFold
        - RepeatedStratifiedKFold
        - RepeatedKFold
        - TimeSeriesSplit

        The function automatically resolves the "auto" strategy and ensures minimum valid
        folds. For classification, it reduces folds if any class has fewer samples than
        the requested number of splits.

        Args:
            target (pd.Series): Target variable used to determine class distribution
                and validate classification constraints.
            resolved (ResolvedConfig): Fully resolved experiment configuration containing
                CV settings, problem type, and runtime parameters.

        Returns:
            Union[StratifiedKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit]:
                Configured cross-validation object ready for use in model evaluation.

        Raises:
            ValueError: If classification target has fewer than 2 unique classes.
            ValueError: If stratified CV is requested for non-classification tasks.
            ValueError: If an unknown CV strategy is specified.
    """
    # --- Aliases for readability ---
    task = resolved.problem.task
    cv = resolved.cv
    runtime = resolved.runtime

    cv_strategy = cv.strategy
    n_splits = cv.folds
    n_repeats = cv.repeats
    seed = runtime.seed

    # --- Validate classification target ---
    if task == "classification" and target.nunique() < 2:
        raise ValueError("Classification requires at least 2 classes.")

    # --- Normalize strategy ---
    if cv_strategy == "auto":
        cv_strategy = "stratified" if task == "classification" else "kfold"

    # --- Stratified ---
    if cv_strategy == "stratified":
        if task != "classification":
            raise ValueError("Stratified CV only works for classification.")

        min_class_count = target.value_counts().min()
        effective_folds = max(2, min(n_splits, int(min_class_count)))

        if effective_folds != n_splits:
            warnings.warn(
                f"Reducing CV folds from {n_splits} to {effective_folds} due to small class size."
            )

        return StratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=seed,
        )

    # --- KFold ---
    if cv_strategy == "kfold":
        return KFold(
            n_splits=max(2, n_splits),
            shuffle=True,
            random_state=seed,
        )

    # --- Repeated ---
    if cv_strategy == "repeated":
        if task == "classification":
            min_class_count = target.value_counts().min()
            effective_folds = max(2, min(n_splits, int(min_class_count)))

            if effective_folds != n_splits:
                warnings.warn(
                    f"Reducing CV folds from {n_splits} to {effective_folds} due to small class size."
                )

            return RepeatedStratifiedKFold(
                n_splits=effective_folds,
                n_repeats=n_repeats,
                random_state=seed,
            )

        return RepeatedKFold(
            n_splits=max(2, n_splits),
            n_repeats=n_repeats,
            random_state=seed,
        )

    # --- Time Series ---
    if cv_strategy == "timeseries":
        return TimeSeriesSplit(
            n_splits=max(2, n_splits)
        )

    # --- Unknown ---
    raise ValueError(f"Unknown CV strategy: {cv_strategy}")
