# core/metadata.py
from dataclasses import dataclass


@dataclass
class DataInfo:
    n_rows: int
    n_features: int

    has_categorical: bool
    is_sparse: bool

    has_missing: bool = False
    missing_fraction: float = 0.0

    n_classes: int | None = None
    minority_ratio: float | None = None

    max_cardinality: int = 0
