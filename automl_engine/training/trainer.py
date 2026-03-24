# training/trainer.py

from typing import Any, Tuple

import pandas as pd

from automl_engine.evaluation import scout_models
from automl_engine.reporting import print_section

from .workflow import execute_training_workflow
from ..planning.experiment.resolved import ResolvedConfig
from ..runtime.state import AutoMLState


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig
) -> tuple[Any, AutoMLState, list[float], str, Any]:
    """
    Execute the full AutoML training workflow.

    Returns:
        Tuple:
            - Final tuned & fitted pipeline
            - AutoMLState (leaderboard state)
            - Outer CV scores
            - Best model name
            - Optional Optuna plot dictionary
    """

    # ---------- Scout ----------
    if resolved.runtime.log:
        print_section("Global Pre-Screen")

    scout_models(
        X,
        y,
        resolved,
    )

    # ---------- Nested Evaluation + Selection + Optimization ----------
    final_pipeline, state, outer_scores, best_model_name, optuna_plots = execute_training_workflow(
        X,
        y,
        resolved
    )

    return final_pipeline, state, outer_scores, best_model_name, optuna_plots
