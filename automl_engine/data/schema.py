# data/schema.py

import pandas as pd
import pandas.api.types as ptypes
from typing import Optional, Set
from automl_engine.planning.config import TaskType

__all__ = [
    "infer_target",
    "COMMON_TARGET_NAMES",
]

COMMON_TARGET_NAMES: Set[str] = {"target", "label", "y", "class"}


def infer_target(df: pd.DataFrame, explicit_target: Optional[str] = None) -> str:
    """
    Infer the target column name from a DataFrame.
    """
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target '{explicit_target}' not found.")
        return explicit_target

    for col in df.columns:
        if col.lower() in COMMON_TARGET_NAMES:
            return col

    return df.columns[-1]


def infer_task(y: pd.Series) -> TaskType:

    if (
        ptypes.is_string_dtype(y)
        or ptypes.is_object_dtype(y)
        or ptypes.is_categorical_dtype(y)
        or ptypes.is_bool_dtype(y)
    ):
        return "classification"

    if ptypes.is_numeric_dtype(y):
        unique_vals: int = y.nunique(dropna=True)

        if unique_vals <= 20:
            return "classification"

        if ptypes.is_float_dtype(y):
            return "regression"

        if ptypes.is_integer_dtype(y):
            return "regression"

    raise ValueError(f"Unable to infer task from target dtype: {y.dtype}")
