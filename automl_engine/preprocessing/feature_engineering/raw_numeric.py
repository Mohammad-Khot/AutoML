# automl_engine/preprocessing/feature_engineering/raw_numeric.py

import numpy as np
import pandas as pd

from .base import BaseFeatureEngineer


class RawNumericFE(BaseFeatureEngineer):
    def __init__(self, resolved, model_spec):
        self.resolved = resolved
        self.model_spec = model_spec
        self.num_cols_ = None
        self.create_log_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            return self

        cfg = self.resolved.feature_generation
        method = cfg.method

        if method in ("none", None):
            return self

        self.num_cols_ = X.select_dtypes(include="number").columns.tolist()
        self.create_log_ = {}

        for col in self.num_cols_:
            self.create_log_[col] = (X[col] > 0).all()

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        cfg = self.resolved.feature_generation
        method = cfg.method
        strategy = cfg.strategy

        if method in ("none", None) or self.num_cols_ is None:
            return X

        X = X.copy()
        created = 0
        max_features = cfg.max_generated_features

        new_features = {}

        # SAFE / MINIMAL
        if strategy in ("safe", "minimal", "auto"):
            for col in self.num_cols_:
                if created >= max_features:
                    break

                new_features[f"{col}_sq"] = X[col] ** 2
                created += 1

                if created >= max_features:
                    break

                if self.create_log_.get(col, False):
                    new_features[f"{col}_log"] = np.log1p(X[col])
                    created += 1

        # POLYNOMIAL
        if method == "polynomial":
            degree = cfg.max_polynomial_degree

            for col in self.num_cols_:
                for d in range(2, degree + 1):
                    if created >= max_features:
                        break
                    new_features[f"{col}_pow{d}"] = X[col] ** d
                    created += 1

        # INTERACTIONS
        if method in ("interactions", "auto"):
            cols = self.num_cols_

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if created >= max_features:
                        break
                    new_features[f"{cols[i]}_x_{cols[j]}"] = (
                        X[cols[i]] * X[cols[j]]
                    )
                    created += 1

        # SINGLE CONCAT → fixes fragmentation
        if new_features:
            new_df = pd.DataFrame(new_features, index=X.index)
            X = pd.concat([X, new_df], axis=1)

        return X
