# utils/seed.py
import random
import numpy as np

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    return None


def inject_seed(estimator, seed):
    if seed is not None:
        try:
            params = estimator.get_params(deep=False)
            if "random_state" in params:
                estimator.set_params(random_state=seed)
        except ValueError as e:
            raise ValueError(
                f"Failed to set random_state on {type(estimator).__name__}"
            ) from e
