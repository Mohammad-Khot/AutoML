# training/trainer.py

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from automl_engine.runtime.state import AutoMLState
from automl_engine.evaluation import scout_models
from automl_engine.reporting import print_section

from .workflow import execute_training_workflow
from .selection import resolve_best_model
from .finalizer import finalize_model
from .. import AutoMLConfig


class ModelTrainer:
    """
    Orchestrates the full AutoML training lifecycle:
    scouting, evaluation, model selection, and final fitting.
    """

    def __init__(self, config: AutoMLConfig, seed: int) -> None:
        """
        Initialize the trainer.

        Args:
            config (AutoMLConfig): Configuration controlling training behavior.
            seed (int): Random seed for reproducibility.
        """
        self.config: AutoMLConfig = config
        self.seed: int = seed

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Dict[str, Any]],
        outer_cv: BaseCrossValidator,
        resolved: Dict[str, Any],
    ) -> Tuple[BaseEstimator, AutoMLState, Dict[str, Any], str]:
        """
        Execute the full training workflow.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            models (Dict[str, Dict[str, Any]]): Model configurations.
            outer_cv (BaseCrossValidator): Cross-validation strategy.
            resolved (Dict[str, Any]): Resolved hyperparameters or metadata.

        Returns:
            Tuple[BaseEstimator, AutoMLState, Dict[str, Any], str]:
                - Final fitted pipeline
                - AutoML state containing evaluation details
                - Outer cross-validation scores
                - Name of the selected best model
        """
        # ---------- Scout ----------
        if self.config.log:
            print_section("Global Pre-Screen")

        models, _ = scout_models(
            X,
            y,
            models,
            outer_cv,
            self.config,
        )

        # ---------- Evaluation ----------
        state, outer_scores = execute_training_workflow(
            X,
            y,
            models,
            outer_cv,
            self.config,
            resolved,
        )

        # ---------- Selection ----------
        best_model_name: str = resolve_best_model(state)

        # ---------- Final Fit ----------
        pipeline: BaseEstimator = finalize_model(
            best_model_name,
            state,
            models,
            X,
            y,
            self.config,
            self.seed,
        )

        return pipeline, state, outer_scores, best_model_name
