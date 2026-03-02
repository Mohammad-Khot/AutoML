# core/__init__.py

from automl_engine.runtime.state import AutoMLState
from automl_engine.planning.models.selector import is_model_suitable
from automl_engine.planning.metadata import DataInfo

__all__ = [
    "select_best_model",
    "MODEL_REGISTRY",
    "MODEL_PRIORITY",
    "AutoMLState",
    "is_model_suitable",
    "DataInfo"
]
