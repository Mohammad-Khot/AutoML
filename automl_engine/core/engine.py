# core/engine.py

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from automl_engine.data import infer_target, load_table
from automl_engine.utils import set_global_seed, save_object
from automl_engine.training.trainer import ModelTrainer
from automl_engine.planning.experiment.resolver import ExperimentResolver
from automl_engine.runtime.session import TrainingSession
from automl_engine.reporting import (
    print_section,
    print_result_block,
    print_run_header,
    print_row,
)


class AutoMLEngine:
    """
    Orchestrates the full AutoML lifecycle including experiment resolution,
    model training, evaluation, reporting, prediction, and persistence.

    The engine is single-use per instance. After calling `fit`, the engine
    stores a TrainingSession containing the resolved configuration, best
    pipeline, search state, outer CV scores, and metadata required for
    prediction and reporting.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the AutoML engine.

        Parameters
        ----------
        config : Any
            Configuration object containing experiment settings.
        """
        self._runtime: Optional[float] = None
        self.config: Any = config

        self.seed: int = (
            np.random.randint(0, 10_000)
            if config.seed is None
            else config.seed
        )

        self.session: Optional[TrainingSession] = None

    # ==========================================================
    # Public Fit APIs
    # ==========================================================

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            save_dir: Optional[str] = None,
    ) -> None:
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

        feature_names: list[str] = list(X.columns)
        outer_cv = resolved.cv_object
        models = resolved.models

        trainer = ModelTrainer(self.config, self.seed)

        (
            final_pipeline,
            state,
            outer_scores,
            best_model_name,
            optuna_plots,
        ) = trainer.train(X, y, models, outer_cv, resolved)

        self.session = TrainingSession(
            resolved=resolved,
            pipeline=final_pipeline,
            search_state=state,
            outer_scores=outer_scores,
            best_model_name=best_model_name,
            feature_names=feature_names,
            optuna_plots=optuna_plots,
        )

        self._runtime = time.perf_counter() - start_time

        if save_dir:
            self._persist(save_dir)

    def fit_from_df(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Fit the engine from a full DataFrame containing features and target.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset containing features and target column.
        target : Optional[str], default=None
            Name of the target column. If None, it is inferred.
        save_dir : Optional[str], default=None
            Directory to persist artifacts.
        """
        if self.fitted:
            raise RuntimeError("Engine has already been fitted.")

        if target is None:
            target = infer_target(df, self.config.target)

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        X: pd.DataFrame = df.drop(columns=[target])
        y: pd.Series = df[target]

        self.fit(X, y, save_dir=save_dir)

    def fit_from_path(
        self,
        path: str | Path,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Load dataset from file path and fit the engine.

        Parameters
        ----------
        path : str
            Path to the dataset file.
        save_dir : Optional[str], default=None
            Directory to persist artifacts.
        """
        df: pd.DataFrame = load_table(path)
        self.fit_from_df(df, save_dir=save_dir)

    # ==========================================================
    # Prediction / Reporting
    # ==========================================================

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained best pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix matching training schema.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
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

    def leaderboard(self, sort: bool = True) -> pd.DataFrame:
        """
        Return a leaderboard of models ranked by mean validation score.

        Parameters
        ----------
        sort : bool, default=True
            Whether to sort models by descending score.

        Returns
        -------
        pd.DataFrame
            DataFrame containing model names and mean scores.
        """
        if not self.fitted:
            raise RuntimeError("Engine not trained yet.")

        inner_scores: Dict[str, float] = self.session_.search_state.scores
        df = pd.DataFrame.from_dict(
            inner_scores,
            orient="index",
            columns=["Mean Score"],
        )

        if sort and "Mean Score" in df.columns:
            df = df.sort_values("Mean Score", ascending=False)

        return df

    def outer_summary(self) -> Optional[Dict[str, float]]:
        """
        Return summary statistics of outer cross-validation scores.

        Returns
        -------
        Optional[Dict[str, float]]
            Dictionary containing folds, mean, and std if nested CV is enabled.
        """
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

    def summary(self) -> None:
        """
        Print leaderboard and nested CV summary (if available).
        """
        if not self.fitted:
            raise RuntimeError("Engine not trained yet.")

        print_section("Leaderboard")
        df = self.leaderboard()

        for name, score in df["Mean Score"].items():
            print_row(name, f"{score:.4f}")

        outer = self.outer_summary()

        state_scores = self.session_.search_state.scores
        best = self.session_.best_model_name
        mean_score = state_scores[best]

        print_result_block(
            model=best,
            metric=self.resolved.metric,
            mean=mean_score,
            std=0.0,  # or compute CV std if you store it later
            runtime=getattr(self, "_runtime", 0.0),
        )

        # ---- Optuna Plots ----
        if getattr(self.config, "show_optuna_plots", False):
            plots = self.session_.optuna_plots
            if plots:
                for name, fig in plots.items():
                    print_section(f"Optuna Plot: {name}")
                    fig.show()

    # ==========================================================
    # Persistence
    # ==========================================================

    def _persist(self, save_dir: str) -> None:
        """
        Persist training session and configuration to disk.

        Parameters
        ----------
        save_dir : str
            Directory path for saving artifacts.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        save_object(self.session_, save_path / "session.joblib")
        save_object(self.config, save_path / "config.joblib")

    # ==========================================================
    # State Management
    # ==========================================================

    def _reset_state(self) -> None:
        """
        Reset engine state by clearing the training session.
        """
        self.session = None

    @property
    def resolved(self) -> Any:
        """
        Access resolved experiment configuration.

        Returns
        -------
        Any
            Resolved configuration object.
        """
        if self.session is None:
            raise RuntimeError("Engine not fitted yet.")
        return self.session.resolved

    @property
    def fitted(self) -> bool:
        """
        Whether the engine has been fitted.

        Returns
        -------
        bool
            True if fitted, otherwise False.
        """
        return self.session is not None

    @property
    def session_(self) -> TrainingSession:
        """
        Access the current training session.

        Returns
        -------
        TrainingSession
            Active training session.
        """
        if not self.fitted:
            raise RuntimeError("Engine not fitted yet.")
        return self.session
