# evaluation/metrics.py

from typing import Callable, Optional
from sklearn.metrics import get_scorer


CLASSIFICATION_METRICS: dict[str, str] = {
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc_ovr",
    "precision": "precision",
    "recall": "recall",
}

REGRESSION_METRICS: dict[str, str] = {
    "r2": "r2",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}

DEFAULT_METRIC: dict[str, str] = {
    "classification": "accuracy",
    "regression": "r2",
}


def get_scorer_safe(metric_name: str) -> Callable:
    """
    Safely retrieve a scikit-learn scorer by name.

    Parameters
    ----------
    metric_name : str
        The name of the scoring metric compatible with sklearn.

    Returns
    -------
    Callable
        A scorer callable object.

    Raises
    ------
    ValueError
        If the provided metric name is not recognized by sklearn.
    """
    try:
        return get_scorer(metric_name)
    except ValueError as exc:
        raise ValueError(f"Unknown metric: {metric_name}") from exc


def resolve_metric(task: str, metric: Optional[str]) -> str:
    """
    Resolve and validate a metric name based on the task type.

    Parameters
    ----------
    task : str
        The type of ML task ("classification" or "regression").
    metric : Optional[str]
        User-specified metric name. If None, the default metric for the task is used.

    Returns
    -------
    str
        The corresponding sklearn-compatible metric string.

    Raises
    ------
    ValueError
        If the task type or metric name is invalid.
    """
    if task not in DEFAULT_METRIC:
        raise ValueError(f"Unknown task: {task}")

    if metric is None:
        return DEFAULT_METRIC[task]

    metric = metric.lower()

    if task == "classification":
        if metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"Invalid classification metric '{metric}'. "
                f"Allowed: {list(CLASSIFICATION_METRICS)}"
            )
        return CLASSIFICATION_METRICS[metric]

    if task == "regression":
        if metric not in REGRESSION_METRICS:
            raise ValueError(
                f"Invalid regression metric '{metric}'. "
                f"Allowed: {list(REGRESSION_METRICS)}"
            )
        return REGRESSION_METRICS[metric]

    raise ValueError(f"Unknown task: {task}")
