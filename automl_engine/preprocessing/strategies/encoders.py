# preprocessing/encoders.py
from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import TransformerMixin
from automl_engine.planning.experiment.resolved import ResolvedConfig

ENCODERS: dict[str, callable] = {
    "onehot": lambda: OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ),
    "ordinal": lambda: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
}


def select_encoder_strategy(resolved: ResolvedConfig):

    strategy = resolved.preprocessing.encoding_strategy
    data_info = resolved.artifacts.data_info

    if strategy == "none":
        return None

    if strategy not in ENCODERS and strategy != "auto":
        raise ValueError(f"Unknown encoding strategy: {strategy}")

    max_unique = data_info.max_cardinality

    if strategy == "auto":
        if max_unique <= resolved.preprocessing.max_cardinality_one_hot:
            return ENCODERS["onehot"]()
        else:
            return ENCODERS["ordinal"]()

    return ENCODERS[strategy]()
