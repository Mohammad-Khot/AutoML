# preprocessing/imputer.py

from typing import Optional, Tuple

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator

from automl_engine.planning.experiment.resolved import ResolvedConfig


def select_imputer_strategy(
    resolved: ResolvedConfig,
) -> Tuple[Optional[BaseEstimator], Optional[BaseEstimator]]:
    """Select numeric and categorical imputation strategies using DataInfo."""
    data_info = resolved.artifacts.data_info
    strategy = resolved.preprocessing.imputation_strategy
    add_indicator = resolved.preprocessing.add_missing_indicator

    if not data_info.has_missing or strategy == "none":
        return None, None

    num_strategy: Optional[BaseEstimator] = None
    cat_strategy: Optional[BaseEstimator] = None

    if strategy == "simple":
        num_strategy = SimpleImputer(strategy="median", add_indicator=add_indicator)
        if data_info.has_categorical:
            cat_strategy = SimpleImputer(
                strategy="most_frequent",
                add_indicator=add_indicator,
            )

    elif strategy == "knn":
        if data_info.n_rows > 20000:
            num_strategy = SimpleImputer(
                strategy="median",
                add_indicator=add_indicator,
            )
        else:
            num_strategy = KNNImputer(add_indicator=add_indicator)

        if data_info.has_categorical:
            cat_strategy = SimpleImputer(
                strategy="most_frequent",
                add_indicator=add_indicator,
            )

    elif strategy == "iterative":
        if data_info.n_rows > 5000:
            num_strategy = SimpleImputer(
                strategy="median",
                add_indicator=add_indicator,
            )
        else:
            num_strategy = IterativeImputer(
                random_state=resolved.runtime.seed,
                add_indicator=add_indicator,
            )

        if data_info.has_categorical:
            cat_strategy = SimpleImputer(
                strategy="most_frequent",
                add_indicator=add_indicator,
            )

    elif strategy == "auto":
        if data_info.n_rows < 2000:
            num_strategy = IterativeImputer(
                random_state=resolved.runtime.seed,
                add_indicator=add_indicator,
            )
        elif data_info.n_rows < 20000:
            num_strategy = KNNImputer(add_indicator=add_indicator)
        else:
            num_strategy = SimpleImputer(
                strategy="median",
                add_indicator=add_indicator,
            )

        if data_info.has_categorical:
            cat_strategy = SimpleImputer(
                strategy="most_frequent",
                add_indicator=add_indicator,
            )

    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    return num_strategy, cat_strategy
