# data/leakage.py

import pandas as pd
import numpy as np
from typing import List, Tuple

from scipy.stats import spearmanr

from automl_engine.planning.config import DataQualityConfig

import warnings
warnings.simplefilter("always")

# ─────────────── Constants ───────────────

LEAK_DUPLICATE = "duplicate_of_target"
LEAK_ID = "possible_identifier"
LEAK_TEMPORAL = "temporal_suspect"
LEAK_TEMPORAL_DTYPE = "temporal_suspect (datetime dtype)"

TEMPORAL_KEYWORDS = [
    "date", "time", "ts", "year", "month",
    "created", "updated", "timestamp"
]


# ─────────────── Detection ───────────────

def detect_id_columns(X: pd.DataFrame, threshold: float) -> List[Tuple[str, str]]:
    """
    Detect columns that behave like identifiers based on uniqueness ratio.

    Columns with a high ratio of unique values relative to dataset size
    are likely identifiers (e.g., user_id, transaction_id) and may cause
    data leakage or poor generalization.

    Args:
        X (pd.DataFrame): Feature matrix.
        threshold (float): Minimum unique ratio to consider a column as an identifier.

    Returns:
        List[Tuple[str, str]]: List of (column_name, reason) tuples indicating detected signals.

    Raises:
        ValueError: If threshold is not between 0 and 1.
    """
    signals: List[Tuple[str, str]] = []
    df_size: int = max(1, len(X))

    for col in X.columns:
        if (
                pd.api.types.is_float_dtype(X[col])
                or pd.api.types.is_datetime64_any_dtype(X[col])
        ):
            continue

        unique_ratio: float = X[col].nunique(dropna=True) / df_size

        if unique_ratio > threshold:
            signals.append((col, LEAK_ID))

    return signals


def detect_target_leakage(X: pd.DataFrame, y: pd.Series) -> List[Tuple[str, str]]:
    """
        Detect features that directly or indirectly leak target information.

        Identifies exact duplicates, near-perfect linear correlations, and
        strong monotonic relationships between features and the target.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.

        Returns:
            List[Tuple[str, str]]: List of (column_name, reason) tuples indicating leakage.

        Raises:
            Exception: If correlation computation fails for a column.
    """
    signals: List[Tuple[str, str]] = []

    for col in X.columns:

        if X[col].equals(y):
            signals.append((col, LEAK_DUPLICATE))
            continue

        if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(y):

            paired_data: pd.DataFrame = pd.concat([X[col], y], axis=1).dropna()

            if len(paired_data) < 2:
                continue

            if paired_data.iloc[:, 0].nunique() <= 1:
                continue

            if paired_data.iloc[:, 1].nunique() <= 1:
                continue

            try:
                corr: float = paired_data.iloc[:, 0].corr(paired_data.iloc[:, 1])
                rho, _ = spearmanr(paired_data.iloc[:, 0], paired_data.iloc[:, 1])

                if abs(rho) > 0.98:
                    signals.append((col, f"near_perfect_monotonic_relation ({rho:.2f})"))

                elif np.isfinite(corr) and abs(corr) > 0.98:
                    signals.append((col, f"near_perfect_linear_corr ({corr:.2f})"))

            except Exception as e:
                signals.append((col, f"analysis_failed: {type(e).__name__}"))

    return signals


def detect_temporal_columns(X: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Detect columns that may represent temporal information.

    Flags datetime dtype columns and columns with names suggesting
    temporal meaning (e.g., 'date', 'timestamp').

    Args:
        X (pd.DataFrame): Feature matrix.

    Returns:
        List[Tuple[str, str]]: List of (column_name, reason) tuples indicating temporal signals.

    Raises:
        None
    """
    signals: List[Tuple[str, str]] = []

    for col in X.columns:

        if pd.api.types.is_datetime64_any_dtype(X[col]):
            signals.append((col, LEAK_TEMPORAL_DTYPE))

        if isinstance(col, str):
            col_str = col.lower()
            if any(k in col_str for k in TEMPORAL_KEYWORDS):
                signals.append((col, LEAK_TEMPORAL))

    return signals


def collect_leakage_signals(X: pd.DataFrame, y: pd.Series, dq_config: DataQualityConfig) -> List[Tuple[str, str]]:
    """
        Aggregate all leakage detection signals from multiple detectors.

        Combines identifier detection, target leakage detection, and temporal
        feature detection into a unified signal list.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            dq_config (DataQualityConfig): Configuration for data quality checks.

        Returns:
            List[Tuple[str, str]]: Aggregated leakage signals.

        Raises:
            None
    """
    signals: List[Tuple[str, str]] = []

    signals += detect_id_columns(X, threshold=dq_config.id_threshold)
    signals += detect_target_leakage(X, y)
    signals += detect_temporal_columns(X)

    signals = list(set(signals))

    return signals


# ─────────────── Handling ───────────────

def _warn(leaks: List[Tuple[str, str]]) -> None:
    """
    Emit a formatted warning for detected leakage signals.

    Args:
        leaks (List[Tuple[str, str]]): List of leakage signals.

    Returns:
        None: This function does not return anything.

    Raises:
        None
    """
    message = "\n".join(
        ["Potential data leakage signals detected:"]
        + [f" • {col}: {reason}" for col, reason in leaks]
    )

    warnings.warn(message)


def apply_leakage_policy(
    X: pd.DataFrame,
    y: pd.Series,
    dq_config: DataQualityConfig
) -> tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Apply configured leakage handling policy to the dataset.

    Based on the detected leakage signals, this function either raises
    an error, warns the user, or drops problematic columns.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        dq_config (DataQualityConfig): Configuration specifying handling policy.

    Returns:
        tuple[pd.DataFrame, List[Tuple[str, str]]]:
            - Cleaned feature matrix.
            - List of detected leakage signals.

    Raises:
        ValueError: If leakage is detected and policy is set to 'error'.
    """
    leaks = collect_leakage_signals(X, y, dq_config=dq_config)

    if not leaks:
        return X, []

    policy = dq_config.leak_handling

    if policy == "error":
        raise ValueError("Data leakage detected")

    if policy == "warn":
        _warn(leaks)
        return X, leaks

    if policy == "drop":
        _warn(leaks)

        to_drop = [
            col for col, reason in leaks
            if reason == LEAK_DUPLICATE
        ]

        X = X.drop(columns=to_drop, errors="ignore")
        return X, leaks

    return X, leaks
