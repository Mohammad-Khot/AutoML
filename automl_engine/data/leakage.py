# data/leakage.py

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from automl_engine.planning.config import DataQualityConfig

LEAK_DUPLICATE = "duplicate_of_target"
LEAK_ID = "possible_identifier"
LEAK_TEMPORAL = "temporal_suspect"
LEAK_TEMPORAL_DTYPE = "temporal_suspect (datetime dtype)"

TEMPORAL_KEYWORDS = [
    "date", "time", "ts", "year", "month",
    "created", "updated", "timestamp",
]


def detect_id_columns(X: pd.DataFrame, threshold: float) -> List[Tuple[str, str]]:
    signals: List[Tuple[str, str]] = []
    df_size = max(1, len(X))

    for col in X.columns:
        if (
            pd.api.types.is_float_dtype(X[col])
            or pd.api.types.is_datetime64_any_dtype(X[col])
        ):
            continue

        unique_ratio = X[col].nunique(dropna=True) / df_size
        if unique_ratio > threshold:
            signals.append((col, LEAK_ID))

    return signals


def detect_target_leakage(X: pd.DataFrame, y: pd.Series) -> List[Tuple[str, str]]:
    signals: List[Tuple[str, str]] = []

    for col in X.columns:
        if X[col].equals(y):
            signals.append((col, LEAK_DUPLICATE))
            continue

        if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(y):
            paired_data = pd.concat([X[col], y], axis=1).dropna()

            if len(paired_data) < 2:
                continue
            if paired_data.iloc[:, 0].nunique() <= 1:
                continue
            if paired_data.iloc[:, 1].nunique() <= 1:
                continue

            try:
                corr = paired_data.iloc[:, 0].corr(paired_data.iloc[:, 1])
                rho, _ = spearmanr(paired_data.iloc[:, 0], paired_data.iloc[:, 1])

                if np.isfinite(rho) and abs(rho) > 0.98:
                    signals.append((col, f"near_perfect_monotonic_relation ({rho:.2f})"))
                elif np.isfinite(corr) and abs(corr) > 0.98:
                    signals.append((col, f"near_perfect_linear_corr ({corr:.2f})"))

            except Exception as exc:
                signals.append((col, f"analysis_failed: {type(exc).__name__}"))

    return signals


def detect_temporal_columns(X: pd.DataFrame) -> List[Tuple[str, str]]:
    signals: List[Tuple[str, str]] = []

    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            signals.append((col, LEAK_TEMPORAL_DTYPE))

        if isinstance(col, str):
            col_str = col.lower()
            if any(keyword in col_str for keyword in TEMPORAL_KEYWORDS):
                signals.append((col, LEAK_TEMPORAL))

    return signals


def collect_leakage_signals(
    X: pd.DataFrame,
    y: pd.Series,
    dq_config: DataQualityConfig,
) -> List[Tuple[str, str]]:
    signals: List[Tuple[str, str]] = []
    signals += detect_id_columns(X, threshold=dq_config.id_threshold)
    signals += detect_target_leakage(X, y)
    signals += detect_temporal_columns(X)

    # Preserve detector order while removing duplicates so warnings and tests
    # are deterministic across Python processes.
    return list(dict.fromkeys(signals))


def _warn(leaks: List[Tuple[str, str]]) -> None:
    message = "\n".join(
        ["Potential data leakage signals detected:"]
        + [f" • {col}: {reason}" for col, reason in leaks]
    )
    warnings.warn(message, stacklevel=2)


def apply_leakage_policy(
    X: pd.DataFrame,
    y: pd.Series,
    dq_config: DataQualityConfig,
) -> tuple[pd.DataFrame, List[Tuple[str, str]]]:
    leaks = collect_leakage_signals(X, y, dq_config=dq_config)

    if not leaks:
        return X, []

    policy = dq_config.leak_handling

    if policy == "error":
        details = ", ".join(f"{col}: {reason}" for col, reason in leaks)
        raise ValueError(f"Data leakage detected: {details}")

    if policy == "warn":
        _warn(leaks)
        return X, leaks

    if policy == "drop":
        _warn(leaks)
        to_drop = [col for col, reason in leaks if reason == LEAK_DUPLICATE]
        return X.drop(columns=to_drop, errors="ignore"), leaks

    raise ValueError(f"Unknown leakage handling policy: {policy}")
