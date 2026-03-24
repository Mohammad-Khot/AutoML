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

    # ───────── (X, y) case ─────────
    if isinstance(data, tuple):
        X, y = data
        return X, y

    # ───────── Load dataframe ─────────
    elif isinstance(data, pd.DataFrame):
        df = data

    elif isinstance(data, (str, Path)):
        df = load_table(data)

    else:
        raise TypeError(
            "data must be one of: (X,y), DataFrame, or file path."
        )

    # ───────── Target handling ─────────
    target = infer_target(df, config.problem.target)

    # ───────── Split ─────────
    X = df.drop(columns=[target])
    y = df[target]

    return X, y
