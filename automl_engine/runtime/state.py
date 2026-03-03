# runtime/state.py

import math
from typing import Any, Dict, List, Optional, Tuple


class AutoMLState:
    """
    Maintains the state of evaluated models during an AutoML run.

    Stores model scores along with their associated pipeline objects
    and hyperparameters. Provides utilities to retrieve scores,
    best-performing models, and sorted rankings.
    """

    def __init__(self) -> None:
        """
        Initialize an empty AutoML state container.
        """
        self.models: Dict[str, Dict[str, Any]] = {}

    def update(
        self,
        model_name: str,
        score: float,
        pipeline: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add or update a model entry in the state.

        Args:
            model_name: Name of the model.
            score: Evaluation score for the model.
            pipeline: Trained pipeline object (optional).
            params: Hyperparameters used for the model (optional).

        Raises:
            TypeError: If score is not numeric.
            ValueError: If score is NaN.
        """
        if not isinstance(score, (int, float)):
            raise TypeError("score must be numeric")

        if math.isnan(score):
            raise ValueError("score cannot be NaN")

        self.models[model_name] = {
            "score": float(score),
            "pipeline": pipeline,
            "params": params,
        }

    @property
    def scores(self) -> Dict[str, float]:
        """
        Retrieve a mapping of model names to their scores.

        Returns:
            Dictionary mapping model names to float scores.
        """
        return {
            name: info["score"]
            for name, info in self.models.items()
        }

    def get_pipeline(self, model_name: str) -> Any:
        """
        Retrieve the stored pipeline for a given model.

        Args:
            model_name: Name of the model.

        Returns:
            The stored pipeline object, or None if not found.
        """
        return self.models.get(model_name, {}).get("pipeline")

    def get_params(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the stored hyperparameters for a given model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary of parameters if present, otherwise None.
        """
        return self.models.get(model_name, {}).get("params")

    def best(self) -> Optional[str]:
        """
        Get the name of the best-performing model.

        Returns:
            Model name with the highest score, or None if no models exist.
        """
        return max(self.scores, key=self.scores.get) if self.scores else None

    def as_sorted(self) -> List[Tuple[str, float]]:
        """
        Retrieve model scores sorted in descending order.

        Returns:
            List of (model_name, score) tuples sorted by score descending.
        """
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
