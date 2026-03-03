# data/schema.py

import pandas as pd
from typing import Optional, Set

COMMON_TARGET_NAMES: Set[str] = {"target", "label", "y", "class"}


def infer_target(df: pd.DataFrame, explicit_target: Optional[str] = None) -> str:
    """
    Infer the target column name from a DataFrame.

    If an explicit target is provided, it validates its presence in the DataFrame.
    Otherwise, it searches for common target column names (case-insensitive).
    If none are found, it defaults to the last column in the DataFrame.

    :param df: Input DataFrame containing features and target.
    :param explicit_target: Optional explicit target column name.
    :return: Inferred target column name.
    :raises ValueError: If the explicit target is provided but not found.
    """
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target '{explicit_target}' not found.")
        return explicit_target

    for col in df.columns:
        if col.lower() in COMMON_TARGET_NAMES:
            return col

    return df.columns[-1]


def infer_task(y: pd.Series) -> str:
    """
    Infer the machine learning task type from a target Series.

    Determines whether the task is classification or regression
    based on the dtype and distribution of unique values.

    :param y: Target variable as a pandas Series.
    :return: Task type as either "classification" or "regression".
    :raises ValueError: If the task cannot be inferred from the dtype.
    """
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        unique_vals: int = y.nunique(dropna=True)

        if unique_vals == 2:
            return "classification"

        if pd.api.types.is_float_dtype(y):
            return "regression"

        if pd.api.types.is_integer_dtype(y):
            unique_set = set(y.unique())
            if unique_set.issubset(set(range(unique_vals))):
                return "classification"
            return "regression"

        return "regression"

    raise ValueError("Unable to infer task from target dtype")
