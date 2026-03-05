# planning/experiment/resolved.py

from dataclasses import dataclass
from typing import Any, Dict, Optional
from automl_engine.planning.config import TaskType


@dataclass
class ResolvedConfig:
    """
    Container for all fully resolved AutoML configuration artifacts.

    This object is created after task inference, metric resolution,
    model suitability filtering, cross-validation construction,
    and leakage checks have been completed.

    Attributes
    ----------
    task : str
        The inferred machine learning task (e.g., "classification", "regression").
    metric : str
        The resolved evaluation metric used for model selection.
    models : Dict[str, Any]
        Dictionary mapping model names to their instantiated objects.
    cv_object : Any
        Cross-validation strategy object (e.g., KFold, StratifiedKFold).
    data_info : Any
        Metadata describing dataset properties (shape, feature types, etc.).
    label_encoder : Optional[Any]
        Label encoder used for classification targets, if applicable.
    leaks : Any
        Results from data leakage checks.
    """
    task: TaskType
    metric: str
    models: Dict[str, Any]
    cv_object: Any
    data_info: Any
    label_encoder: Optional[Any]
    leaks: Any
