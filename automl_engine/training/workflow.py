# training/workflow.py
from typing import Any, Dict, Tuple

import pandas as pd

from automl_engine.evaluation import evaluate_models
from automl_engine.orchestration.nested import run_nested_cv
from automl_engine.reporting import print_section
from automl_engine import AutoMLConfig


def execute_training_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Dict[str, Any]],
    outer_cv: Any,
    config: AutoMLConfig,
    resolved: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    # noinspection GrazieInspection
    """
        Execute the training workflow using either standard cross-validation
        or nested cross-validation based on configuration.

        If nested CV is disabled, models are evaluated using standard
        cross-validation and the resulting state and scores are returned.

        If nested CV is enabled, nested evaluation is performed to obtain
        unbiased outer scores, followed by a final model fit on the full
        dataset. The final fitted state and outer scores are returned.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            models (Dict[str, Dict[str, Any]]): Dictionary of model configurations.
            outer_cv (Any): Cross-validation splitter for outer evaluation.
            config (AutoMLConfig): AutoML configuration object.
            resolved (Dict[str, Any]): Resolved configuration or search space.

        Returns:
            Tuple[Any, Dict[str, Any]]:
                - state: Final evaluation state object.
                - scores: Cross-validation scores (outer scores if nested CV).
        """
    if not config.nested_cv:

        if config.log:
            print_section("Standard Cross Validation")

        state = evaluate_models(
            X,
            y,
            models,
            outer_cv,
            config,
            resolved,
            "OUTER_CV",
        )

        return state, state.scores

    # ---------- Nested ----------
    if config.log:
        print_section("Nested Evaluation")

    outer_result = run_nested_cv(
        X,
        y,
        models,
        outer_cv,
        config,
        resolved,
    )

    outer_scores = getattr(
        outer_result,
        "scores",
        outer_result,
    )

    if config.log:
        print_section("Final Model Fit")

    state = evaluate_models(
        X,
        y,
        models,
        outer_cv,
        config,
        resolved,
        "FINAL_FIT",
    )

    return state, outer_scores
