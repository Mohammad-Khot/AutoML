# utils/persistence.py

from pathlib import Path
from typing import Any

import joblib


def save_pipeline(pipeline: Any, path: Path) -> None:
    """
    Serialize and save a trained pipeline object to the specified file path.

    Ensures that the parent directory exists before saving.

    Args:
        pipeline: The trained pipeline object to persist.
        path: Destination file path where the pipeline will be saved.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path) -> Any:
    """
    Load and deserialize a pipeline object from the specified file path.

    Args:
        path: File path from which the pipeline will be loaded.

    Returns:
        The deserialized pipeline object.
    """
    return joblib.load(path)


def save_object(obj: Any, path: Path) -> None:
    """
    Serialize and save a generic Python object to the specified file path.

    Ensures that the parent directory exists before saving.

    Args:
        obj: The Python object to persist.
        path: Destination file path where the object will be saved.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path) -> Any:
    """
    Load and deserialize a generic Python object from the specified file path.

    Args:
        path: File path from which the object will be loaded.

    Returns:
        The deserialized Python object.
    """
    return joblib.load(path)
