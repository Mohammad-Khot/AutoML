from sklearn.base import BaseEstimator, TransformerMixin

from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.planning.models import ModelSpec


class BaseFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, resolved: ResolvedConfig, model_spec: ModelSpec):
        self.resolved = resolved
        self.model_spec = model_spec

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
