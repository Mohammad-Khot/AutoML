# utils/seed.py

import os
import random
import numpy as np
from typing import Optional, TypeVar, Any
from sklearn.base import BaseEstimator

T = TypeVar("T", bound=BaseEstimator)


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for full reproducibility across Python,
    NumPy, and hashing behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)


def inject_seed(estimator: T, seed: Optional[int]) -> T:
    """
    Inject a random_state into a scikit-learn estimator if supported.
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
