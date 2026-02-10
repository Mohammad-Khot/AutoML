# core/engine.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from automl_engine.data import load_csv, infer_target, infer_task, run_leakage_checks
from automl_engine.preprocessing import build_pipeline
from automl_engine.utils import set_global_seed, save_pipeline, save_object
from automl_engine.optimization import evaluate_models, filter_by_dummy_once
from automl_engine.evaluation import get_cv, nested_cv, resolve_metric
from automl_engine.core import MODEL_REGISTRY, MODEL_PRIORITY, select_best_model


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

        # ----------Data Loading----------
        df = load_csv(csv_path)
        target = infer_target(df, self.config.target)

        self.leaks = run_leakage_checks(df, target)

        X = df.loc[:, df.columns != target]
        y = df[target]

        # ----------Task Inference----------
        task = self.config.task or infer_task(y)
        self.config.task = task

        if task == "classification":
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y),
                          index=df.index)

        # ----------Metric Resolution----------
        self.config.metric = resolve_metric(task, self.config.metric)

        # ----------Dataset Sanity----------
        if task == "classification":
            min_class = y.value_counts().min()

            if min_class < max(2, self.config.cv_folds):
                raise ValueError(
                    f"Not enough samples per class for {self.config.cv_folds}-fold CV. "
                    f"Smallest class has {min_class} samples."
                )

        # ----------CV Setup----------
        outer_cv = get_cv(task, y, self.config.cv_folds, self.seed)
        models = MODEL_REGISTRY[task]

        # ----------Model Filtering----------
        if self.config.allowed_models:
            models = {
                name: info
                for name, info in models.items()
                if name in self.config.allowed_models
            }
            if not models:
                raise ValueError("allowed_models filtered out all models")

        # ----------Pre-filter before nested----------
        print("\n=== GLOBAL PRE-SCREEN ===")
        models = filter_by_dummy_once(X, y, models, outer_cv, self.config)

        if len(models) <= 1:
            print("[WARN] Only dummy model remains after filtering.")

        # ----------Evaluation----------
        print("\n=== RUNNING NESTED EVALUATION ===")

        if self.config.nested_cv:
            outer_scores = nested_cv(X, y, models, outer_cv, self.config)
        else:
            print("Nested CV Disabled - using standard cross-validation.")
            outer_scores = {}

        # ----------Final Training----------
        print("\n=== FINAL MODEL SELECTION ===")

        state = evaluate_models(X, y, models, outer_cv, self.config)

        best_model_name = select_best_model(state.scores, MODEL_PRIORITY)
        best_info = models[best_model_name]

        best_pipeline = build_pipeline(best_info, X, self.config)
        best_pipeline.fit(X, y)

        # ----------Persistence----------
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            save_pipeline(best_pipeline,  save_dir/"model.joblib")
            save_object(state.scores, save_dir/"scores.joblib")

            if self.label_encoder:
                save_object(self.label_encoder, save_dir/"label_encoder.joblib"
                            )
            save_object(self.config, save_dir/"config.joblib")

        return best_pipeline, {
            "inner_scores": state.scores,
            "outer_scores": outer_scores,
            "nested_enabled": self.config.nested_cv
        }
