# core/metadata.py
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class DataInfo:
    n_rows: int
    n_features: int

    has_categorical: bool
    is_sparse: bool

    has_missing: bool = False
    missing_fraction: float = 0.0

    n_classes: Optional[int] = None
    minority_ratio: Optional[float] = None

    max_cardinality: int = 0

    @classmethod
    def from_data(cls, X, y=None) -> "DataInfo":

        # --- Shape ---
        n_rows = X.shape[0]
        n_features = X.shape[1]

        # --- Sparsity ---
        is_sparse = sparse.isspmatrix(X) or getattr(X, "sparse", False)

        # --- Categorical detection (DataFrame only) ---
        has_categorical = False
        max_cardinality = 0

        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["object", "category"])
            has_categorical = len(cat_cols.columns) > 0

            if has_categorical:
                max_cardinality = max(
                    X[col].nunique(dropna=True) for col in cat_cols.columns
                )

        # --- Missing values ---
        has_missing = False
        missing_fraction = 0.0

        if isinstance(X, pd.DataFrame):
            total_cells = n_rows * n_features
            missing_count = X.isna().sum().sum()
            has_missing = missing_count > 0
            missing_fraction = missing_count / total_cells if total_cells > 0 else 0.0
        else:
            # numpy case
            if not is_sparse:
                missing_count = np.isnan(X).sum()
                total_cells = X.size
                has_missing = missing_count > 0
                missing_fraction = missing_count / total_cells if total_cells > 0 else 0.0

        # --- Target analysis ---
        n_classes = None
        minority_ratio = None

        if y is not None:
            y_array = np.asarray(y)

            # classification check
            unique_classes, counts = np.unique(y_array, return_counts=True)

            if len(unique_classes) > 1:
                n_classes = len(unique_classes)
                minority_ratio = counts.min() / counts.sum()

        return cls(
            n_rows=n_rows,
            n_features=n_features,
            has_categorical=has_categorical,
            is_sparse=is_sparse,
            has_missing=has_missing,
            missing_fraction=missing_fraction,
            n_classes=n_classes,
            minority_ratio=minority_ratio,
            max_cardinality=max_cardinality,
        )
