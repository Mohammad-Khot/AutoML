# training/trainer.py

from typing import Any

import pandas as pd

from automl_engine.evaluation import scout_models
from automl_engine.reporting import print_section

from .workflow import execute_training_workflow
from ..planning.experiment.resolved import ResolvedConfig
from ..runtime.state import AutoMLState


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig,
) -> tuple[Any, AutoMLState, list[float], str, Any]:
    """Execute scouting followed by the selected training workflow."""
    if resolved.runtime.log:
        print_section("Global Pre-Screen")

    selected_models, _ = scout_models(X, y, resolved)

    # Scouting is a real pre-screen, not just logging. Restrict the expensive
    # workflow to the selected ModelSpec objects so top_k_models has effect.
    resolved.artifacts.models = dict(selected_models)

    return execute_training_workflow(X, y, resolved)
