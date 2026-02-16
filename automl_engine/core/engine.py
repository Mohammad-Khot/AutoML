# core/engine.py

from pathlib import Path
from scipy import sparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from automl_engine.core.registry import COST_LOW, COST_MEDIUM
from automl_engine.data import load_table, infer_target, infer_task, run_leakage_checks
from automl_engine.utils import set_global_seed, save_pipeline, save_object
from automl_engine.evaluation import get_cv_object, resolve_metric
from automl_engine.core import MODEL_REGISTRY, is_model_suitable, DataInfo
from automl_engine.training.trainer import ModelTrainer


class AutoMLEngine:
    def __init__(self, config):
        self.config = config

        self.seed = (
            np.random.randint(0, 10_000)
            if config.seed is None
            else config.seed
        )

        self.leaks = None
        self.label_encoder = None

    def run(self, csv_path: str, save_dir: str | None = None) -> tuple:
        set_global_seed(self.seed)

        X, y, task = self._prepare_data(csv_path)

        data_info = DataInfo(
            n_rows=X.shape[0],
            n_features=X.shape[1],
            has_categorical=X.select_dtypes(include=["object", "category"]).shape[1] > 0,
            is_sparse=sparse.isspmatrix(X) or getattr(X, "sparse", False),
        )

        outer_cv = get_cv_object(task, y, self.config.cv_folds, self.seed)

        models = dict(MODEL_REGISTRY[task])
        models = self._filter_models(models, data_info)

        trainer = ModelTrainer(self.config, self.seed)

        best_pipeline, state, outer_scores = trainer.train(
            X, y, models, outer_cv, task
        )

        if save_dir:
            self._persist(save_dir, best_pipeline, state, outer_scores)

        return best_pipeline, {
            "inner_scores": state.scores,
            "outer_scores": outer_scores,
        }

    def _prepare_data(self, csv_path: str):
        df = load_table(csv_path)
        target = infer_target(df, self.config.target)

        self.leaks = run_leakage_checks(df, target)

        X = df.loc[:, df.columns != target]
        y = df[target]

        task = self.config.task or infer_task(y)
        if task not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported task: {task}")

        self.config.task = task

        if task == "classification" and y.dtype in ("object", "category"):
            self.label_encoder = LabelEncoder()
            y = pd.Series(
                self.label_encoder.fit_transform(y),
                index=df.index
            )

        self.config.metric = resolve_metric(task, self.config.metric)

        return X, y, task

    def _filter_models(self, models, data_info):
        if self.config.allowed_models:
            models = {
                name: info
                for name, info in models.items()
                if name in self.config.allowed_models
            }
            if not models:
                raise ValueError("allowed_models filtered out all models")

        models = {
            name: info
            for name, info in models.items()
            if is_model_suitable(name, info, data_info)
        }

        if not models:
            raise ValueError("All models rejected by suitability rules.")

        if self.config.max_compute == "low":
            models = {n: i for n, i in models.items() if i["compute_cost"] == COST_LOW}
        elif self.config.max_compute == "medium":
            models = {
                n: i
                for n, i in models.items()
                if i["compute_cost"] in (COST_LOW, COST_MEDIUM)
            }

        if not models:
            raise ValueError(
                f"No models available under compute level: {self.config.max_compute}"
            )

        return models

    def _persist(self, save_dir, best_pipeline, state, outer_scores):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_pipeline(best_pipeline, save_dir / "model.joblib")
        save_object(state.scores, save_dir / "scores.joblib")

        if self.label_encoder:
            save_object(self.label_encoder, save_dir / "label_encoder.joblib")

        save_object(self.config, save_dir / "config.joblib")

        print(state.scores)
        print(outer_scores)
