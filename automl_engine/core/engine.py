# core/engine.py

from pathlib import Path
import numpy as np
import pandas as pd

from automl_engine.data import (
    infer_target,
    load_table,
)
from automl_engine.utils import (
    set_global_seed,
    save_object,
)

from automl_engine.training.trainer import ModelTrainer
from automl_engine.core.resolver import ExperimentResolver
from automl_engine.core.session import TrainingSession
from automl_engine.utils.run_loggerr import print_run_header
from automl_engine.utils.console import print_section, print_result_block, CONSOLE_WIDTH
from automl_engine.utils.table import print_row


class AutoMLEngine:
    # ==========================================================
    # Initialization
    # ==========================================================
    def __init__(self, config):
        self._runtime = None
        self.config = config

        self.seed = (
            np.random.randint(0, 10_000)
            if config.seed is None
            else config.seed
        )
        self.session = None

    # ==========================================================
    # Public Fit APIs
    # ==========================================================
    def fit(self, X: pd.DataFrame, y: pd.Series, save_dir: str | None = None) -> None:
        import time
        start_time = time.perf_counter()

        if self.fitted:
            raise RuntimeError("Engine has already been fitted.")

        set_global_seed(self.seed)

        resolver = ExperimentResolver(self.config, self.seed)

        X, y, resolved = resolver.resolve(X, y)

        if self.config.log:
            print_run_header(
                task=resolved.task,
                metric=resolved.metric,
                n_samples=len(X),
                n_features=X.shape[1],
                models=resolved.models,
                seed=self.seed,
                cv=resolved.cv_object,
                search_type=self.config.search_type,
            )

        feature_names = list(X.columns)

        outer_cv = resolved.cv_object
        models = resolved.models

        trainer = ModelTrainer(self.config, self.seed)

        best_pipeline, state, outer_scores, best_model_name = trainer.train(
            X, y, models, outer_cv, resolved
        )

        self.session = TrainingSession(
            resolved=resolved,
            pipeline=best_pipeline,
            search_state=state,
            outer_scores=outer_scores,
            best_model_name=best_model_name,
            feature_names=feature_names,
        )

        runtime = time.perf_counter() - start_time
        self._runtime = runtime

        if save_dir:
            self._persist(save_dir)

    def fit_from_df(
            self,
            df: pd.DataFrame,
            target: str | None = None,
            save_dir: str | None = None,
    ):
        if self.fitted:
            raise RuntimeError("Engine has already been fitted.")

        if target is None:
            target = infer_target(df, self.config.target)

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        X = df.drop(columns=[target])
        y = df[target]

        return self.fit(X, y, save_dir=save_dir)

    def fit_from_path(self, path: str, save_dir: str | None = None):
        df = load_table(path)
        return self.fit_from_df(df, save_dir=save_dir)

    # ==========================================================
    # Prediction / Reporting
    # ==========================================================
    def predict(self, X: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Engine must be trained before prediction.")

        if list(X.columns) != self.session_.feature_names:
            raise ValueError(
                "Input feature order/schema differs from training data."
            )

        pipeline = self.session_.pipeline
        preds = pipeline.predict(X)

        encoder = self.resolved.label_encoder
        if encoder is not None:
            preds = encoder.inverse_transform(preds)

        return preds

    def leaderboard(self, sort=True) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Engine not trained yet.")

        inner_scores = self.session_.search_state.scores
        df = pd.DataFrame.from_dict(inner_scores, orient="index", columns=["Mean Score"])

        if sort and "Mean Score" in df.columns:
            df = df.sort_values("Mean Score", ascending=False)

        return df

    def outer_summary(self):
        if not self.fitted:
            raise RuntimeError("Engine not trained yet.")

        if not self.config.nested_cv or self.session_.outer_scores is None:
            return None

        scores = self.session_.outer_scores

        return {
            "folds": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    def summary(self):
        if not self.fitted:
            raise RuntimeError("Engine not trained yet.")

        print_section("Leaderboard")

        df = self.leaderboard()

        for name, score in df["Mean Score"].items():
            print_row(name, f"{score:.4f}")

        outer = self.outer_summary()
        if outer and self.config.log:
            print_result_block(
                model=self.session_.best_model_name,
                metric=self.resolved.metric,
                mean=outer["mean"],
                std=outer["std"],
                runtime=getattr(self, "_runtime", 0.0),
            )

    # ==========================================================
    # Persistence
    # ==========================================================
    def _persist(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_object(
            self.session_,
            save_dir / "session.joblib"
        )

        save_object(
            self.config,
            save_dir / "config.joblib"
        )

    # ==========================================================
    # State Management
    # ==========================================================
    def _reset_state(self):
        self.session = None

    @property
    def resolved(self):
        if self.session is None:
            raise RuntimeError("Engine not fitted yet.")
        return self.session.resolved

    @property
    def fitted(self):
        return self.session is not None

    @property
    def session_(self):
        if not self.fitted:
            raise RuntimeError("Engine not fitted yet.")
        return self.session
