import pandas as pd
import numpy as np
import warnings
from typing import List, Tuple, Any

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr


def detect_id_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[Any, str]]:
    suspects = []
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            continue
        unique_ratio = df[col].nunique(dropna=True) / max(1, len(df))
        if unique_ratio > threshold:
            suspects.append((col, "possible_identifier"))
    return suspects


def detect_target_duplicates(df: pd.DataFrame, target: str) -> List[Tuple[Any, str]]:
    suspects = []
    y = df[target]

    for col in df.drop(columns=[target]):
        if df[col].equals(y):
            suspects.append((col, "duplicate_of_target"))

        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                valid = df[[col, target]].dropna()

                if len(valid) < 2:
                    continue

                if valid[col].nunique() == 1:
                    continue

                corr = np.corrcoef(valid[col], valid[target])[0, 1]
                rho, _ = spearmanr(valid[col], valid[target])
                mi = mutual_info_regression(valid[[col]], valid[target])[0]

                if abs(corr) > 0.98:
                    suspects.append((col, f"near_perfect_linear_corr ({corr:.2f})"))
                elif abs(rho) > 0.98:
                    suspects.append((col, f"near_perfect_monotonic_relation ({rho:.2f})"))
                elif mi > 0.98:
                    suspects.append((col, f"strong_nonlinear_dependency ({mi:.2f})"))

            except Exception:
                continue

    return suspects


def detect_temporal_columns(df: pd.DataFrame) -> List[Tuple[Any, str]]:
    suspects = []
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            suspects.append((col, "temporal_suspect"))
    return suspects


def run_leakage_checks(df: pd.DataFrame, target: str) -> List[Tuple[Any, str]]:
    report = []

    report += detect_id_columns(df)
    report += detect_target_duplicates(df, target)
    report += detect_temporal_columns(df)

    if not report:
        return []

    lines = ["Potential data leakage signals detected:"]
    for col, reason in report:
        lines.append(f" • {col}: {reason}")

    warnings.warn("\n".join(lines))
    return report
