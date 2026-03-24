import numpy as np
import pandas as pd

from .base import BaseFeatureEngineer


class VectorFE(BaseFeatureEngineer):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cfg = self.resolved.feature_generation

        if cfg.method in ("none", None):
            return X

        if isinstance(X, pd.DataFrame):
            X = X.values

        max_features = cfg.max_generated_features

        n_samples, n_features = X.shape
        created = 0
        new_features = []

        # ───────── INTERACTIONS (VECTOR LEVEL) ─────────
        if cfg.method in ("interactions", "auto"):
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if created >= max_features:
                        break

                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    new_features.append(interaction)
                    created += 1

        if new_features:
            new_block = np.hstack(new_features)
            return np.hstack([X, new_block])

        return np.asarray(X)
