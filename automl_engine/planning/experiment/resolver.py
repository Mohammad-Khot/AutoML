# core/resolver.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from automl_engine.data import (
    infer_task,
    run_leakage_checks,
)

from automl_engine.evaluation import (
    resolve_metric,
    get_cv_object,
)

from automl_engine.planning.models.registry import (
    MODEL_REGISTRY,
    COST_LOW,
    COST_MEDIUM,
)

from automl_engine.planning.models.selector import (
    is_model_suitable,
)

from automl_engine.planning.metadata import DataInfo

from automl_engine.planning.experiment.resolved import ResolvedConfig


class ExperimentResolver:

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def resolve(self, X: pd.DataFrame, y: pd.Series):

        X = X.copy()
        y = y.copy()

        # -------------------------
        # Leakage
        # -------------------------
        leaks = run_leakage_checks(X, y)

        # -------------------------
        # Task inference
        # -------------------------
        task = self.config.task or infer_task(y)

        # -------------------------
        # Label encoding
        # -------------------------
        label_encoder = None
        if task == "classification" and not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = pd.Series(
                label_encoder.fit_transform(y),
                index=y.index
            )

        # -------------------------
        # Metric resolution
        # -------------------------
        metric = resolve_metric(task, self.config.metric)

        # -------------------------
        # Data info
        # -------------------------
        data_info = DataInfo.from_data(X, y)

        # -------------------------
        # Models
        # -------------------------
        models = dict(MODEL_REGISTRY[task])
        models = self._filter_models(models, data_info)

        # -------------------------
        # CV
        # -------------------------
        cv_object = get_cv_object(
            task,
            y,
            self.config.cv_folds,
            self.seed
        )

        resolved = ResolvedConfig(
            task=task,
            metric=metric,
            models=models,
            cv_object=cv_object,
            data_info=data_info,
            label_encoder=label_encoder,
            leaks=leaks,
        )

        return X, y, resolved

    # -------------------------
    # Model filtering moved here
    # -------------------------
    def _filter_models(self, models, data_info):

        cfg = self.config

        if cfg.allowed_models:
            models = {
                name: info for name, info in models.items()
                if name in cfg.allowed_models
            }

        models = {
            name: info
            for name, info in models.items()
            if is_model_suitable(name, info, data_info)
        }

        if cfg.max_compute == "low":
            models = {
                name: info for name, info in models.items()
                if info["compute_cost"] == COST_LOW
            }

        elif cfg.max_compute == "medium":
            models = {
                name: info for name, info in models.items()
                if info["compute_cost"] in (COST_LOW, COST_MEDIUM)
            }

        if not models:
            raise ValueError("No models available after filtering.")

        return models
