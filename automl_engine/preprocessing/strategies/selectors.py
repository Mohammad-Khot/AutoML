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

from automl_engine.planning.config import FeatureSelectionMethod, MLTask


# ─────────────── Smart K Selection ───────────────

def compute_adaptive_k(n_features: int) -> int:
    """
    Dynamically determine number of features to keep.

    Rules:
    - Small feature space → keep most
    - Medium → moderate reduction
    - Large → aggressive reduction
    """

    if n_features <= 20:
        return n_features

    if n_features <= 100:
        return max(10, int(0.5 * n_features))

    if n_features <= 500:
        return max(20, int(0.2 * n_features))

    return max(30, int(0.1 * n_features))


# ─────────────── Importance-based fallback ───────────────

def get_tree_based_selector(task: MLTask) -> BaseEstimator:
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    return SelectFromModel(estimator=model, threshold="median")


def get_l1_selector(task: MLTask) -> BaseEstimator:
    if task == "classification":
        base = LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            max_iter=5000,     # ✅ increase a lot
            tol=1e-3,          # ✅ relax convergence
            C=1.0
        )
    else:
        base = Lasso(alpha=0.01)

    return SelectFromModel(estimator=base)


# ─────────────── Main Selector Factory ───────────────

def get_selector(
    task: MLTask,
    mode: FeatureSelectionMethod = "auto",
    n_features: Optional[int] = None,
) -> Union[str, BaseEstimator]:

    if mode == "none" or not n_features:
        return "passthrough"

    # ───────── AUTO MODE (ROBUST) ─────────
    if mode == "auto":

        # tiny feature space → don't touch
        if n_features <= 10:
            return "passthrough"

        k = compute_adaptive_k(n_features)

        # if reduction meaningless → skip
        if k >= n_features:
            return "passthrough"

        # small feature space → mutual info
        if n_features < 50:
            if task == "classification":
                return SelectKBest(mutual_info_classif, k=k)
            return SelectKBest(mutual_info_regression, k=k)

        # ✅ CRITICAL CHANGE:
        # mid + large → tree-based (stable)
        return get_tree_based_selector(task)

    # ───────── MANUAL MODES ─────────

    if mode == "variance":
        return VarianceThreshold(threshold=0.0)

    if mode == "l1":
        if n_features < 5:
            return "passthrough"
        return get_l1_selector(task)  # still allowed manually

    if mode == "tree":
        if n_features < 3:
            return "passthrough"
        return get_tree_based_selector(task)

    raise ValueError(f"Unknown selector mode: {mode}")
