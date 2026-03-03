# utils/seed.py

import random
import numpy as np
from typing import Optional, TypeVar, Any
from sklearn.base import BaseEstimator

T = TypeVar("T", bound=BaseEstimator)


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for Python's built-in random module
    and NumPy to ensure reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to initialize the random number generators.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)


def inject_seed(estimator: T, seed: Optional[int]) -> T:
    """
    Inject a random_state into a scikit-learn estimator if supported.

    This function checks whether the estimator exposes a `random_state`
    parameter and, if so, sets it to the provided seed. If the estimator
    does not support `random_state`, it is returned unchanged.

    Parameters
    ----------
    estimator : BaseEstimator
        A scikit-learn compatible estimator instance.
    seed : Optional[int]
        The seed value to inject. If None, no changes are made.

    Returns
    -------
    T
        The estimator instance with updated random_state if applicable.

    Raises
    ------
    ValueError
        If setting the random_state fails.
    """
    if seed is not None:
        try:
            params: dict[str, Any] = estimator.get_params(deep=False)
            if "random_state" in params:
                estimator.set_params(**{"random_state": seed})
        except ValueError as e:
            raise ValueError(
                f"Failed to set random_state on {type(estimator).__name__}"
            ) from e

    return estimator
