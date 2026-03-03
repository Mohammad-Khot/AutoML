# runtime/session.py

from dataclasses import dataclass
from typing import Any, Sequence
from automl_engine.planning.experiment.resolved import ResolvedConfig


@dataclass
class TrainingSession:
    """
    Container for all artifacts produced during an AutoML training run.

    Attributes:
        resolved (ResolvedConfig): Fully resolved experiment configuration,
            including task type, metric, models, CV object, and metadata.
        pipeline (Any): Final fitted pipeline (including preprocessing
            and model) selected after search.
        search_state (Any): Object containing search results, rankings,
            and intermediate evaluation information.
        outer_scores (Sequence[float]): Scores from outer cross-validation
            folds (used in nested CV scenarios).
        best_model_name (str): Name of the best-performing model selected
            during the search process.
        feature_names (list[str]): List of feature names used during training.
    """

    resolved: ResolvedConfig
    pipeline: Any
    search_state: Any
    outer_scores: Sequence[float]
    best_model_name: str
    feature_names: list[str]
