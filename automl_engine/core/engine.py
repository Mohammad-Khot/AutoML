# core/engine.py

from pathlib import Path
from typing import Optional
import time

import numpy as np
import pandas as pd

from automl_engine import AutoMLConfig
from automl_engine.data.adapter import adapt_input
from automl_engine.planning.experiment import ResolvedConfig, ExperimentResolver
from automl_engine.training.trainer import train_model
from automl_engine.utils import set_global_seed, save_object
from automl_engine.runtime import TrainingSession
from automl_engine.reporting import (
    print_section,
    print_result_block,
    print_run_header,
    print_row,
    print_subsection
)

from automl_engine.preprocessing.describe import describe_pipeline
from automl_engine.reporting.preprocessing import print_preprocessing


class AutoMLEngine:

    def __init__(self, user_config: AutoMLConfig) -> None:
        self._runtime: Optional[float] = None
        self._user_config: AutoMLConfig = user_config
        self._session: Optional[TrainingSession] = None

    def fit(
            self,
            data: pd.DataFrame | tuple[pd.DataFrame, pd.Series] | str | Path,
            save_dir: Optional[str] = None,
    ) -> None:

        start_time = time.perf_counter()

        if self.fitted:
            raise RuntimeError("Engine has already been fitted.")

        set_global_seed(self._user_config.runtime.seed)

        X, y = adapt_input(
            data,
            self._user_config
        )

        resolver = ExperimentResolver(self._user_config)

        X, y, resolved = resolver.resolve(X, y)

        if resolved.runtime.log:
            print_run_header(X, y, resolved)

        feature_names = list(X.columns)

        (
            final_pipeline,
            state,
            outer_scores,
            best_model_name,
            optuna_plots,
        ) = train_model(X, y, resolved)

        if resolved.runtime.log:
            summary = describe_pipeline(final_pipeline)
            print_preprocessing(summary)

        self._session = TrainingSession(
            resolved=resolved,
            pipeline=final_pipeline,
            search_state=state,
            outer_scores=outer_scores,
            best_model_name=best_model_name,
            feature_names=feature_names,
            optuna_plots=optuna_plots,
            label_encoder=resolved.artifacts.label_encoder,
        )

        self._runtime = time.perf_counter() - start_time

        if save_dir:
            self._persist(save_dir)

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        if not self.fitted:
            raise RuntimeError("Engine must be trained before prediction.")

        if list(X.columns) != self.session.feature_names:
            raise ValueError(
                "Input feature order/schema differs from training data."
            )

        pipeline = self.session.pipeline
        preds = pipeline.predict(X)

        encoder = self.session.label_encoder
        if encoder is not None:
            preds = encoder.inverse_transform(preds)

        return preds

    def summary(
            self,
            show_leaderboard: bool = True,
    ) -> None:

        if not self.fitted:
            raise RuntimeError("Engine not trained yet.")

        state_scores = self.session.search_state.scores

        if show_leaderboard:

            print_section("Leaderboard")

            sorted_models = sorted(
                state_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for name, score in sorted_models:
                print_row(name, f"{score:.4f}")

        best = self.session.best_model_name

        if self.resolved.cv.use_nested_cv and self.session.outer_scores:
            from automl_engine.utils import compute_bootstrap_ci

            scores = self.session.outer_scores

            mean, margin = compute_bootstrap_ci(
                scores,
                confidence=0.95,
                n_bootstrap=1000,
                seed=self.resolved.runtime.seed
            )

            std = margin
            label = "Performance (Nested CV)"

        else:
            mean = state_scores[best]
            std = None
            label = "Performance (CV)"

        print_result_block(
            model=self.session.best_model_name,
            metric=self.resolved.problem.metric,
            mean=float(mean),
            std=float(std) if std is not None else None,
            runtime=float(self._runtime or 0.0),
            label=label
        )

        if self._user_config.generate_optuna_plots:

            plots = self.session.optuna_plots

            if plots:
                for name, fig in plots.items():
                    print_subsection(f"Optuna Plot: {name}")
                    fig.show()

    def _persist(self, save_dir: str) -> None:

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        save_object(self.session, save_path / "session.joblib")
        save_object(self._user_config, save_path / "config.joblib")

    @property
    def resolved(self) -> ResolvedConfig:

        if not self.fitted:
            raise RuntimeError("Engine not fitted yet.")

        return self._session.resolved

    @property
    def fitted(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> TrainingSession:

        if not self.fitted:
            raise RuntimeError("Engine not fitted yet.")

        return self._session
