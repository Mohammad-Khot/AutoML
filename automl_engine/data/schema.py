# data/schema.py
import pandas as pd


def infer_target(df, explicit_target=None) -> str:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target '{explicit_target}' not found.")
        return explicit_target
    return df.columns[-1]  # fallback, logged


def infer_task(y) -> str:
    if pd.api.types.is_categorical_dtype(y):
        return "classification"
    if pd.api.types.is_object_dtype(y):
        return "classification"
    if pd.api.types.is_string_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
