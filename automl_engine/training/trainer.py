# training/trainer.py

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from automl_engine.evaluation import scout_models
from automl_engine.reporting import print_section

from .workflow import execute_training_workflow
from .. import AutoMLConfig
from ..planning.experiment.resolved import ResolvedConfig
from ..runtime.state import AutoMLState


class ModelTrainer:
    """
    Orchestrates the full AutoML training lifecycle:
    scouting, nested evaluation, selection, optimization, and final fitting.
    """

    def __init__(self, config: AutoMLConfig, seed: int) -> None:
        self.config: AutoMLConfig = config
        self.seed: int = seed

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Dict[str, Any]],
        outer_cv: BaseCrossValidator,
        resolved: ResolvedConfig,
    ) -> tuple[Any, AutoMLState, list[float], str, Any]:
        """
        Execute the full training workflow.

        Returns:
            Tuple:
                - Final tuned & fitted pipeline
                - AutoMLState (leaderboard state)
                - Outer CV scores
                - Best model name
                - Optional Optuna plot dictionary
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

        # ---------- Nested Evaluation + Selection + Optimization ----------
        final_pipeline, state, outer_scores, best_model_name, optuna_plots = execute_training_workflow(
            X,
            y,
            models,
            outer_cv,
            self.config,
            resolved,
        )

        return final_pipeline, state, outer_scores, best_model_name, optuna_plots
