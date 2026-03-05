# planning/models/policy.py
from typing import Any, Dict, List

from automl_engine.planning.models.spec import ModelSpec

TOL: float = 1e-6


def select_best_model(
    scores: Dict[str, Any],
    models: Dict[str, ModelSpec],
) -> str:
    """
    Select the best model based on the highest score with deterministic tie-breaking.

    Supports two score formats:
    1. Flat format:
        {"model_a": 0.91, "model_b": 0.89}
    2. Nested format:
        {"model_a": {"mean_score": 0.91}, "model_b": {"mean_score": 0.89}}

    Tie-breaking order:
    1. Highest score
    2. Model priority (lower value indicates higher priority)
    3. Lexicographic order of the model name
    """

    if not scores:
        raise ValueError("No model could be evaluated successfully.")

    first_value: Any = next(iter(scores.values()))

    if isinstance(first_value, dict):
        max_score: float = max(v["mean_score"] for v in scores.values())
        tied: List[str] = [
            name for name, v in scores.items()
            if abs(v["mean_score"] - max_score) <= TOL
        ]
    else:
        max_score = max(scores.values())
        tied = [
            name for name, score in scores.items()
            if abs(score - max_score) <= TOL
        ]

    return min(
        tied,
        key=lambda name: (
            models[name].priority,
            name,
        ),
    )
