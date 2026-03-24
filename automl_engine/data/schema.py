# data/schema.py

import pandas as pd
import pandas.api.types as ptypes
from typing import Optional

from automl_engine.planning.config import MLTask

_COMMON_TARGET_NAMES: set[str] = {"target", "label", "y", "output"}


def infer_target(df: pd.DataFrame, target_name: Optional[str] = None) -> str:
    """
    Infer and validate the target column from a DataFrame.

    The function determines the target column using the following priority:
    1. If an explicit target is provided, validate and return it.
    2. Search for common target column names (case-insensitive).
    3. Fallback to the last column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target.
        target_name (Optional[str]): User-specified target column name.

    Returns:
        str: Name of the resolved target column.

    Raises:
        ValueError: If the explicit target is not found in the DataFrame.
        ValueError: If the DataFrame has no columns.
    """
    # ───────── Explicit target ─────────
    if target_name is not None:
        if target_name not in df.columns:
            raise ValueError(f"Target '{target_name}' not found in dataframe.")
        return target_name

    # ───────── Common names ─────────
    for column_name in df.columns:
        if isinstance(column_name, str) and column_name.lower() in _COMMON_TARGET_NAMES:
            return column_name

    # ───────── Fallback ─────────
    if df.columns.empty:
        raise ValueError("Empty dataframe — no columns available for target.")

    return df.columns[-1]


def infer_task(target: pd.Series) -> MLTask:
    """
    Infer whether a machine learning task is classification or regression.

    Determines the task type based on the dtype and distribution of the target
    variable. Categorical-like types (string, object, boolean, categorical)
    are treated as classification. Numeric types are classified based on the
    number of unique values relative to dataset size.

    Args:
        target (pd.Series): Target variable from the dataset.

    Returns:
        MLTask: Either "classification" or "regression".

    Raises:
        ValueError: If the target dtype is unsupported for inference.
    """
    # ───────── float types → Regression ─────────

    if ptypes.is_float_dtype(target):
        return "regression"

    # ───────── Categorical-like types → Classification ─────────
    if (
        ptypes.is_string_dtype(target)
        or ptypes.is_object_dtype(target)
        or ptypes.is_bool_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        return "classification"

    # ───────── Numeric types → Heuristic ─────────
    if ptypes.is_numeric_dtype(target):
        num_unique = target.nunique(dropna=True)
        total = len(target)

        ratio = num_unique / total

        if num_unique <= 20 or ratio < 0.01:
            return "classification"

        return "regression"

    # ───────── Unsupported dtype ─────────
    raise ValueError(f"Unable to infer task from target dtype: {target.dtype}")
