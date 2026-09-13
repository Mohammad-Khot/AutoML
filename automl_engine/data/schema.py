# data/schema.py

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from typing import Optional

from automl_engine.planning.config import MLTask

_COMMON_TARGET_NAMES: set[str] = {"target", "label", "y", "output"}


def infer_target(df: pd.DataFrame, target_name: Optional[str] = None) -> str:
    """Infer and validate the target column from a DataFrame."""
    if target_name is not None:
        if target_name not in df.columns:
            raise ValueError(f"Target '{target_name}' not found in dataframe.")
        return target_name

    for column_name in df.columns:
        if isinstance(column_name, str) and column_name.lower() in _COMMON_TARGET_NAMES:
            return column_name

    if df.columns.empty:
        raise ValueError("Empty dataframe — no columns available for target.")

    return df.columns[-1]


def infer_task(target: pd.Series) -> MLTask:
    """Infer whether a target represents classification or regression."""
    if target.empty:
        raise ValueError("Cannot infer task from an empty target.")

    non_null = target.dropna()
    if non_null.empty:
        raise ValueError("Cannot infer task from an all-missing target.")

    if (
        ptypes.is_string_dtype(non_null)
        or ptypes.is_object_dtype(non_null)
        or ptypes.is_bool_dtype(non_null)
        or isinstance(non_null.dtype, pd.CategoricalDtype)
    ):
        return "classification"

    if ptypes.is_numeric_dtype(non_null):
        num_unique = int(non_null.nunique(dropna=True))
        total = len(non_null)
        ratio = num_unique / total

        # Integer-valued floats are common encodings for class labels (0.0/1.0,
        # 1.0/2.0/3.0). Treat low-cardinality integer-like numeric targets as
        # classification rather than blindly considering every float regression.
        values = non_null.to_numpy(dtype=float, copy=False)
        integer_like = bool(np.isfinite(values).all() and np.allclose(values, np.round(values)))

        if integer_like and (num_unique <= 20 or ratio < 0.01):
            return "classification"

        if not ptypes.is_float_dtype(non_null) and (num_unique <= 20 or ratio < 0.01):
            return "classification"

        return "regression"

    raise ValueError(f"Unable to infer task from target dtype: {target.dtype}")
