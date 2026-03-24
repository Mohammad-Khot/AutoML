# runtime/state.py

import math
from typing import Any, Dict, Optional, cast, TypedDict

from sklearn.base import BaseEstimator


class ModelEntry(TypedDict):
    score: float
    pipeline: Optional[BaseEstimator]
    params: Optional[Dict[str, Any]]


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

        Returns:
            None
        """
        self.models: Dict[str, ModelEntry] = {}

    def update(
            self,
            model_name: str,
            score: float,
            pipeline: Optional[BaseEstimator] = None,
            params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add or update a model entry in the state.

        Args:
            model_name: Name of the model.
            score: Evaluation score for the model.
            pipeline: Trained pipeline object (optional).
            params: Hyperparameters used for the model (optional).

        Returns:
            None

        Raises:
            TypeError: If score is not numeric.
            ValueError: If score is NaN.
        """
        if not isinstance(score, (int, float)):
            raise TypeError("score must be numeric")

        if math.isnan(score):
            print(f"[WARN] {model_name} produced NaN — assigning -inf")
            score = float("-inf")

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
            name: float(info["score"])
            for name, info in self.models.items()
        }

    def get_pipeline(self, model_name: str) -> BaseEstimator:
        pipeline = self.models.get(model_name, {}).get("pipeline")

        if pipeline is None:
            raise KeyError(f"Pipeline not found for model '{model_name}'")

        return cast(BaseEstimator, pipeline)
