# automl_engine/preprocessing/feature_engineering/raw_categorical.py

import pandas as pd

from .base import BaseFeatureEngineer


class RawCategoricalFE(BaseFeatureEngineer):
    def __init__(self, resolved, model_spec):
        self.resolved = resolved
        self.model_spec = model_spec
        self.cat_cols_ = None
        self.rare_maps_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            return self

        cfg = self.resolved.feature_generation

        if cfg.method in ("none", None):
            return self

        self.cat_cols_ = X.select_dtypes(exclude="number").columns.tolist()
        self.rare_maps_ = {}

        for col in self.cat_cols_:
            freq = X[col].value_counts(normalize=True)
            rare = freq[freq < 0.01].index
            self.rare_maps_[col] = set(rare)

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        cfg = self.resolved.feature_generation

        if cfg.method in ("none", None) or self.cat_cols_ is None:
            return X

        X = X.copy()

        for col in self.cat_cols_:
            rare = self.rare_maps_.get(col, set())
            X[col] = X[col].apply(lambda x: "OTHER" if x in rare else x)

        return X
