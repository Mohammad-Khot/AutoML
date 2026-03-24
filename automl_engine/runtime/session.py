# runtime/session.py

from dataclasses import dataclass
from typing import Any

from sklearn.preprocessing import LabelEncoder

from automl_engine.planning.experiment.resolved import ResolvedConfig


@dataclass(slots=True)
class TrainingSession:
    resolved: ResolvedConfig
    pipeline: Any
    search_state: Any
    outer_scores: Any
    best_model_name: str
    feature_names: list[str]
    optuna_plots: dict[str, Any]

    label_encoder: LabelEncoder | None
