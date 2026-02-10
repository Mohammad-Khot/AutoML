# utils/persistence.py

import joblib
from pathlib import Path


def save_pipeline(pipeline, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path):
    return joblib.load(path)


def save_object(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path):
    return joblib.load(path)
