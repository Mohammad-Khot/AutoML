from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from . import infer_target
from .loader import load_table

from .. import AutoMLConfig


DataInput = Union[
    pd.DataFrame,
    Tuple[pd.DataFrame, pd.Series],
    str,
    Path,
]


def adapt_input(
    data: DataInput,
    config: AutoMLConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("Tuple data input must contain exactly (X, y).")

        X, y = data
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if X.empty:
            raise ValueError("Feature matrix X must not be empty.")
        if y.isna().any():
            raise ValueError("Target y contains missing values; target imputation is not supported.")

        # Align by position for CV/pipeline operations while preventing subtle
        # concat/alignment errors in leakage checks when indexes differ.
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return X, y

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, Path)):
        df = load_table(data)
    else:
        raise TypeError("data must be one of: (X, y), DataFrame, or file path.")

    if df.empty:
        raise ValueError("Input dataframe must not be empty.")

    target = infer_target(df, config.problem.target)
    X = df.drop(columns=[target])
    y = df[target]

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if y.isna().any():
        raise ValueError("Target contains missing values; target imputation is not supported.")

    return X.reset_index(drop=True), y.reset_index(drop=True)
