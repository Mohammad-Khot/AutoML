import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from automl_engine import AutoMLConfig

ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore"),
    "ordinal": lambda: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
}


def select_encoder_strategy(X: pd.DataFrame, config: AutoMLConfig):
    cat_cols = X.select_dtypes(exclude="number").columns

    if config.encoding_mode == "none" or len(cat_cols) == 0:
        return "passthrough"

    if config.encoding_mode != "auto":
        return ENCODERS[config.encoding_mode]()

    max_unique = max((X[col].nunique() for col in cat_cols), default=0)

    if max_unique <= config.max_cardinality_one_hot:
        return ENCODERS["onehot"]()

    return ENCODERS["ordinal"]()
