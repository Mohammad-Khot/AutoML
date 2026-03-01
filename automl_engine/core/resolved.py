from dataclasses import dataclass
from sklearn.base import BaseEstimator
from typing import Any

@dataclass
class ResolvedConfig:
    task: str
    metric: str
    models: dict[str, Any]
    cv_object: Any
    data_info: Any
    label_encoder: Any | None
    leaks: Any

