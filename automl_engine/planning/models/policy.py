# planning/models/policy.py

from typing import Any, Dict

TOL: float = 1e-6


def select_best_model(scores: Dict[str, Any], priority: Dict[str, int]) -> str:
    """
    Select the best model based on highest score with deterministic tie-breaking.

    This function supports two score formats:
    1. Flat dictionary format:
       {
           "model_a": 0.91,
           "model_b": 0.89
       }

    2. Nested dictionary format:
       {
           "model_a": {"mean_score": 0.91, ...},
           "model_b": {"mean_score": 0.89, ...}
       }

    The selection process:
    - Identify the maximum score.
    - Collect all models within TOL of the maximum score.
    - Break ties using the provided priority dictionary (lower value = higher priority).
    - If still tied, fall back to lexicographic order of model names.

    Args:
        scores (Dict[str, Any]): Mapping of model names to either float scores
            or dictionaries containing a "mean_score" key.
        priority (Dict[str, int]): Mapping of model names to priority values
            (lower values indicate higher priority).

    Returns:
        str: The name of the selected best model.

    Raises:
        ValueError: If no models are provided in the scores dictionary.
    """
    if not scores:
        raise ValueError("No model could be evaluated successfully.")

    first_value: Any = next(iter(scores.values()))

    if isinstance(first_value, dict):
        max_score: float = max(v["mean_score"] for v in scores.values())
        tied: list[str] = [
            name
            for name, v in scores.items()
            if abs(v["mean_score"] - max_score) <= TOL
        ]
    else:
        max_score = max(scores.values())
        tied = [
            name
            for name, score in scores.items()
            if abs(score - max_score) <= TOL
        ]

    return min(tied, key=lambda name: (priority.get(name, 99), name))
