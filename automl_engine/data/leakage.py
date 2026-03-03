# data/leakage.py

import pandas as pd
import numpy as np
import warnings
from typing import List, Tuple

from scipy.stats import spearmanr


TEMPORAL_KEYWORDS = [
    "date", "time", "ts", "year", "month",
    "created", "updated", "timestamp"
]


def detect_id_columns(X: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str]]:
    """
    Identify columns that may act as identifiers based on high uniqueness ratio.

    Args:
        X: Feature DataFrame.
        threshold: Proportion of unique values required to flag a column.

    Returns:
        A list of tuples containing column names and the reason for suspicion.
    """
    suspects: List[Tuple[str, str]] = []
    df_size: int = max(1, len(X))

    for col in X.columns:
        if pd.api.types.is_float_dtype(X[col]):
            continue

        unique_ratio: float = X[col].nunique(dropna=True) / df_size

        if unique_ratio > threshold:
            suspects.append((col, "possible_identifier"))

    return suspects


def detect_target_leakage(X: pd.DataFrame, y: pd.Series) -> List[Tuple[str, str]]:
    """
    Detect potential target leakage by checking duplication or strong correlations
    between features and the target variable.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        A list of tuples containing column names and the reason for suspicion.
    """
    suspects: List[Tuple[str, str]] = []

    for col in X.columns:

        if X[col].equals(y):
            suspects.append((col, "duplicate_of_target"))
            continue

        if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(y):

            valid: pd.DataFrame = pd.concat([X[col], y], axis=1).dropna()

            if len(valid) < 2:
                continue

            if valid.iloc[:, 0].nunique() <= 1:
                continue

            if valid.iloc[:, 1].nunique() <= 1:
                continue

            try:
                corr: float = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                rho, _ = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])

                if np.isfinite(corr) and abs(corr) > 0.98:
                    suspects.append((col, f"near_perfect_linear_corr ({corr:.2f})"))

                elif abs(rho) > 0.98:
                    suspects.append((col, f"near_perfect_monotonic_relation ({rho:.2f})"))

            except Exception as e:
                suspects.append((col, f"analysis_failed: {type(e).__name__}"))

    return suspects


def detect_temporal_columns(X: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Detect columns that may introduce temporal leakage based on datetime dtype
    or temporal keywords in column names.

    Args:
        X: Feature DataFrame.

    Returns:
        A list of tuples containing column names and the reason for suspicion.
    """
    suspects: List[Tuple[str, str]] = []

    for col in X.columns:

        if pd.api.types.is_datetime64_any_dtype(X[col]):
            suspects.append((col, "temporal_suspect (datetime dtype)"))

        elif any(k in col.lower() for k in TEMPORAL_KEYWORDS):
            suspects.append((col, "temporal_suspect"))

    return suspects


def run_leakage_checks(X: pd.DataFrame, y: pd.Series) -> List[Tuple[str, str]]:
    """
    Execute all leakage detection checks and issue a warning if any signals are found.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        A list of tuples containing column names and reasons for potential leakage.
    """
    report: List[Tuple[str, str]] = []

    report += detect_id_columns(X)
    report += detect_target_leakage(X, y)
    report += detect_temporal_columns(X)

    if not report:
        return []

    lines: List[str] = ["Potential data leakage signals detected:"]
    for col, reason in report:
        lines.append(f" • {col}: {reason}")

    warnings.warn("\n".join(lines))

    return report
