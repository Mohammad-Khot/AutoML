# preprocessing/imputer.py

from typing import Optional, Tuple

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer # noqa
from sklearn.base import BaseEstimator

from automl_engine.planning.experiment.resolved import ResolvedConfig


def select_imputer_strategy(
    resolved: ResolvedConfig,
) -> Tuple[Optional[BaseEstimator], Optional[BaseEstimator]]:
    """
    Select numeric and categorical imputation strategies using DataInfo.
    """

    data_info = resolved.artifacts.data_info
    strategy = resolved.preprocessing.imputation_strategy

    # --- no missing values ---
    if not data_info.has_missing:
        return None, None

    if strategy == "none":
        return None, None

    num_strategy: Optional[BaseEstimator] = None
    cat_strategy: Optional[BaseEstimator] = None

    # --- SIMPLE ---
    if strategy == "simple":
        num_strategy = SimpleImputer(strategy="median")
        if data_info.has_categorical:
            cat_strategy = SimpleImputer(strategy="most_frequent")

    # --- KNN ---
    elif strategy == "knn":
        # avoid KNN on large datasets (expensive)
        if data_info.n_rows > 20000:
            num_strategy = SimpleImputer(strategy="median")
        else:
            num_strategy = KNNImputer()

    # --- ITERATIVE ---
    elif strategy == "iterative":
        # fallback if too large (very slow)
        if data_info.n_rows > 5000:
            num_strategy = SimpleImputer(strategy="median")
        else:
            num_strategy = IterativeImputer(random_state=resolved.runtime.seed)

    # --- AUTO ---
    elif strategy == "auto":
        if data_info.n_rows < 2000:
            num_strategy = IterativeImputer(random_state=resolved.runtime.seed)
        elif data_info.n_rows < 20000:
            num_strategy = KNNImputer()
        else:
            num_strategy = SimpleImputer(strategy="median")

        if data_info.has_categorical:
            cat_strategy = SimpleImputer(strategy="most_frequent")

    return num_strategy, cat_strategy
