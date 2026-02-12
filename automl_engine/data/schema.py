# data/schema.py
import pandas as pd

COMMON_TARGET_NAMES = {"target", "label", "y", "class"}


def infer_target(df, explicit_target=None) -> str:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target '{explicit_target}' not found.")
        return explicit_target

    for col in df.columns:
        if col.lower() in COMMON_TARGET_NAMES:
            return col

    return df.columns[-1]


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        unique_vals = y.nunique(dropna=True)

        # binary
        if unique_vals == 2:
            return "classification"

        # floats strongly imply regression
        if pd.api.types.is_float_dtype(y):
            return "regression"

        # integer heuristic
        if pd.api.types.is_integer_dtype(y):
            # check if values form small contiguous class labels
            if set(y.unique()).issubset(set(range(unique_vals))):
                return "classification"

            return "regression"

        return "regression"

    raise ValueError("Unable to infer task from target dtype")
