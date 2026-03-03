# preprocessing/encoders.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import TransformerMixin
from automl_engine import AutoMLConfig


ENCODERS: dict[str, callable] = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore"),
    "ordinal": lambda: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
}


def select_encoder_strategy(
    X: pd.DataFrame,
    config: AutoMLConfig
) -> TransformerMixin | str:
    """
    Select and return an appropriate categorical encoding strategy.

    The selection logic works as follows:
    - If encoding is disabled ("none") or no categorical columns exist,
      return "passthrough".
    - If a specific encoding mode ("onehot" or "ordinal") is provided,
      return the corresponding encoder instance.
    - If encoding mode is "auto", choose:
        * OneHotEncoder if the maximum cardinality among categorical
          columns is less than or equal to `config.max_cardinality_one_hot`.
        * Otherwise, use OrdinalEncoder.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature dataframe.
    config : AutoMLConfig
        Configuration object containing encoding settings.

    Returns
    -------
    TransformerMixin | str
        A fitted sklearn-compatible encoder instance or the string
        "passthrough" if no encoding is required.
    """
    cat_cols = X.select_dtypes(exclude="number").columns

    if config.encoding_mode == "none" or len(cat_cols) == 0:
        return "passthrough"

    if config.encoding_mode != "auto":
        return ENCODERS[config.encoding_mode]()

    max_unique = max((X[col].nunique() for col in cat_cols), default=0)

    if max_unique <= config.max_cardinality_one_hot:
        return ENCODERS["onehot"]()

    return ENCODERS["ordinal"]()
