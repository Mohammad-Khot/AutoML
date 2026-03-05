# preprocessing/imputer.py
from typing import Optional, Tuple

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator

from automl_engine.planning.experiment.resolved import ResolvedConfig


def select_imputer_strategy(
    X: pd.DataFrame,
    config: ResolvedConfig
) -> Tuple[Optional[BaseEstimator], Optional[BaseEstimator]]:
    """
    Select and configure numeric and categorical imputation strategies
    based on the provided AutoML configuration and dataset size.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    config : AutoMLConfig
        AutoML configuration containing imputation strategy and random seed.

    Returns
    -------
    Tuple[Optional[BaseEstimator], Optional[BaseEstimator]]
        A tuple containing:
        - Numeric imputer (or None if not required)
        - Categorical imputer (or None if not required)
    """
    if config.imputation == "none":
        return None, None

    num_strategy: Optional[BaseEstimator] = None
    cat_strategy: Optional[BaseEstimator] = None

    if config.imputation == "simple":
        num_strategy = SimpleImputer(strategy="median")
        cat_strategy = SimpleImputer(strategy="most_frequent")

    elif config.imputation == "knn":
        num_strategy = KNNImputer()

    elif config.imputation == "iterative":
        num_strategy = IterativeImputer(random_state=config.seed)

    elif config.imputation == "auto":
        n_rows: int = X.shape[0]

        if n_rows < 2000:
            num_strategy = IterativeImputer(random_state=config.seed)
        elif n_rows < 20_000:
            num_strategy = KNNImputer()
        else:
            num_strategy = SimpleImputer(strategy="median")

        cat_strategy = SimpleImputer(strategy="most_frequent")

    return num_strategy, cat_strategy
