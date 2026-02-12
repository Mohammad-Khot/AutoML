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


def detect_id_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str]]:
    suspects = []
    df_size = max(1, len(df))
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            continue
        unique_ratio = df[col].nunique(dropna=True) / df_size
        if unique_ratio > threshold:
            suspects.append((col, "possible_identifier"))
    return suspects


def detect_target_duplicates(df: pd.DataFrame, target: str) -> List[Tuple[str, str]]:
    suspects = []

    for col in df.columns:
        if col == target:
            continue

        if df[col].equals(df[target]):
            suspects.append((col, "duplicate_of_target"))
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                valid = df.loc[:, [col, target]].dropna()

                if len(valid) < 2:
                    continue

                if valid[col].nunique() == 1:
                    continue

                if valid[target].nunique() == 1:
                    continue

                corr = valid[col].corr(valid[target])
                rho, _ = spearmanr(valid[col], valid[target])

                if np.isfinite(corr) and abs(corr) > 0.98:
                    suspects.append((col, f"near_perfect_linear_corr ({corr:.2f})"))
                elif abs(rho) > 0.98:
                    suspects.append((col, f"near_perfect_monotonic_relation ({rho:.2f})"))

            except Exception as e:
                suspects.append((col, f"analysis_failed: {type(e).__name__}"))
                continue

    return suspects


def detect_temporal_columns(df: pd.DataFrame) -> List[Tuple[str, str]]:
    suspects = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            suspects.append((col, "temporal_suspect (datetime dtype)"))
        elif any(k in col.lower() for k in TEMPORAL_KEYWORDS):
            suspects.append((col, "temporal_suspect"))
    return suspects


def run_leakage_checks(df: pd.DataFrame, target: str) -> List[Tuple[str, str]]:
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
