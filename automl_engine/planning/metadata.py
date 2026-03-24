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

    # numeric + stats
    num_numeric_features: int = 0
    has_outliers: bool = False
    scale_range_large: bool = False
    has_skewed_features: bool = False
    has_constant_features: bool = False

    # categorical breakdown
    n_categorical_features: int = 0
    n_high_cardinality_features: int = 0
    n_low_cardinality_features: int = 0

    @classmethod
    def from_data(cls, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataInfo":

        n_rows = int(X.shape[0])
        n_features = int(X.shape[1])

        is_sparse = sparse.isspmatrix(X)

        has_categorical = False
        max_cardinality = 0

        num_numeric_features = 0
        has_outliers = False
        scale_range_large = False
        has_skewed_features = False
        has_constant_features = False

        n_categorical_features = 0
        n_high_cardinality_features = 0
        n_low_cardinality_features = 0

        # =========================
        # SPARSE HANDLING
        # =========================
        has_missing = False
        missing_fraction = 0.0

        if is_sparse:
            num_numeric_features = X.shape[1]

            # missing values in sparse
            if X.nnz > 0:
                has_missing = np.isnan(X.data).any()
                missing_fraction = np.isnan(X.data).sum() / X.data.size
            else:
                has_missing = False
                missing_fraction = 0.0

        # =========================
        # DATAFRAME HANDLING
        # =========================
        elif isinstance(X, pd.DataFrame):

            # --- categorical ---
            cat_cols = X.select_dtypes(include=["object", "category", "string"])
            n_categorical_features = len(cat_cols.columns)
            has_categorical = n_categorical_features > 0

            if has_categorical:
                cardinalities = [int(X[col].nunique(dropna=True)) for col in cat_cols.columns]
                max_cardinality = max(cardinalities)

                for c in cardinalities:
                    if c > 20:
                        n_high_cardinality_features += 1
                    else:
                        n_low_cardinality_features += 1

            # --- numeric ---
            num_cols = X.select_dtypes(include="number")
            num_numeric_features = len(num_cols.columns)

            if num_numeric_features > 0:
                numeric_data = num_cols

                # ===== SAMPLE for performance =====
                sample = numeric_data.sample(
                    min(10000, len(numeric_data)), random_state=42
                )

                # --- OUTLIERS (IQR) ---
                q1 = sample.quantile(0.25)
                q3 = sample.quantile(0.75)
                iqr = q3 - q1

                outlier_mask = (
                    (sample < (q1 - 1.5 * iqr)) |
                    (sample > (q3 + 1.5 * iqr))
                )

                outlier_ratio = outlier_mask.to_numpy().mean()
                has_outliers = outlier_ratio > 0.05

                # --- SCALE RANGE (robust version) ---
                col_min = sample.min()
                col_max = sample.max()

                scale_ratio = (col_max / (col_min.abs() + 1e-9)).replace([np.inf, -np.inf], 1)
                scale_range_large = scale_ratio.max() > 100

                # --- SKEWNESS ---
                skew = sample.skew().abs()
                has_skewed_features = (skew > 1).any()

                # --- CONSTANT FEATURES ---
                has_constant_features = (sample.nunique() <= 1).any()

            # --- Missing ---
            total_cells = n_rows * n_features
            missing_count = int(X.isna().sum().sum())
            has_missing = missing_count > 0
            missing_fraction = missing_count / total_cells if total_cells > 0 else 0.0

        # =========================
        # NUMPY ARRAY HANDLING
        # =========================
        else:
            num_numeric_features = n_features

            if not is_sparse:
                missing_count = int(np.isnan(X).sum())
                total_cells = int(X.size)
                has_missing = missing_count > 0
                missing_fraction = missing_count / total_cells if total_cells > 0 else 0.0

        # =========================
        # TARGET
        # =========================
        n_classes = None
        minority_ratio = None

        if y is not None:
            y_array = np.asarray(y)
            unique_classes, counts = np.unique(y_array, return_counts=True)

            if len(unique_classes) > 1:
                n_classes = int(len(unique_classes))
                minority_ratio = float(counts.min() / counts.sum())

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
            num_numeric_features=num_numeric_features,
            has_outliers=has_outliers,
            scale_range_large=scale_range_large,
            has_skewed_features=has_skewed_features,
            has_constant_features=has_constant_features,
            n_categorical_features=n_categorical_features,
            n_high_cardinality_features=n_high_cardinality_features,
            n_low_cardinality_features=n_low_cardinality_features,
        )
