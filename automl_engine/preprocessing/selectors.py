# preprocessing/selectors.py
from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_regression,
    mutual_info_classif,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression

from automl_engine.planning.config import FeatureSelection, TaskType


def safe_k(total_features: int, k: int) -> int:
    """
    Safely bound the number of selected features.

    Ensures that the returned value is at least 1 and at most
    the total number of available features.

    Parameters
    ----------
    total_features : int
        Total number of features available.
    k : int
        Desired number of features.

    Returns
    -------
    int
        A value between 1 and total_features (inclusive).
    """
    return max(1, min(k, total_features))


def get_selector(
    task: TaskType,
    mode: FeatureSelection = "auto",
    n_features: Optional[int] = None,
) -> Union[str, BaseEstimator]:
    """
    Return an appropriate feature selection strategy based on task and mode.

    Parameters
    ----------
    task : {"classification", "regression"}
        The type of machine learning task.
    mode : {"auto", "none", "variance", "l1", "tree"}, default="auto"
        The feature selection strategy to use.
    n_features : int, optional
        Total number of input features.

    Returns
    -------
    Union[str, BaseEstimator]
        A scikit-learn feature selector instance or "passthrough"
        if no selection should be applied.

    Raises
    ------
    ValueError
        If an unknown selector mode is provided.
    """
    if mode == "none" or not n_features:
        return "passthrough"

    if mode == "auto":
        k = safe_k(n_features, 20)
        if k >= n_features:
            return "passthrough"

        if task == "classification":
            return SelectKBest(score_func=mutual_info_classif, k=k)
        return SelectKBest(score_func=mutual_info_regression, k=k)

    if mode == "variance":
        return VarianceThreshold(threshold=0.0)

    if mode == "l1":
        if n_features < 5:
            return "passthrough"

        if task == "classification":
            base: BaseEstimator = LogisticRegression(
                penalty="l1",
                solver="liblinear",
            )
        else:
            base = Lasso(alpha=0.1)

        return SelectFromModel(estimator=base)

    if mode == "tree":
        if n_features < 3:
            return "passthrough"

        if task == "classification":
            base = RandomForestClassifier(n_estimators=100)
        else:
            base = RandomForestRegressor(n_estimators=100)

        return SelectFromModel(estimator=base)

    raise ValueError(f"Unknown selector mode: {mode}")
