# core/session.py

from dataclasses import dataclass
from typing import Any
from automl_engine.planning.experiment.resolved import ResolvedConfig


@dataclass
class TrainingSession:
    resolved: ResolvedConfig
    pipeline: Any
    search_state: Any
    outer_scores: Any
    best_model_name: str
    feature_names: list[str]