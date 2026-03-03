# preprocessing/selection.py

from automl_engine.runtime.state import AutoMLState
from automl_engine.planning.models import (
    MODEL_PRIORITY,
    select_best_model,
)


def resolve_best_model(state: AutoMLState) -> str:
    """
    Determine and return the best model name based on evaluated scores.

    Args:
        state (AutoMLState): The current AutoML state containing model evaluation scores.

    Returns:
        str: The name of the best-performing model according to score and priority.

    Raises:
        RuntimeError: If no models were successfully evaluated.
    """
    if not state.scores:
        raise RuntimeError(
            "No models successfully evaluated."
        )

    return select_best_model(
        state.scores,
        MODEL_PRIORITY,
    )