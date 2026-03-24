# evaluation/metrics.py

from typing import Callable, Optional
from sklearn.metrics import get_scorer

from automl_engine.planning.config import MLTask, MetricName


CLASSIFICATION_METRICS: dict[MetricName, str] = {
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc_ovr",
    "precision": "precision",
    "recall": "recall",
}

REGRESSION_METRICS: dict[MetricName, str] = {
    "r2": "r2",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}

DEFAULT_METRIC: dict[MLTask, MetricName] = {
    "classification": "accuracy",
    "regression": "r2",
}


def get_scorer_safe(metric_name: str) -> Callable:
    try:
        return get_scorer(metric_name)
    except ValueError as exc:
        raise ValueError(f"Unknown sklearn scorer: {metric_name}") from exc


def resolve_metric(task: MLTask, metric: Optional[MetricName]) -> MetricName:
    """
    Resolve and validate the evaluation metric for a given ML task.

    This function determines the appropriate metric to use based on the task type
    (classification or regression). If no metric is provided, a default metric is
    selected. It also ensures that the chosen metric is compatible with the task.

    Args:
        task (MLTask): The machine learning task type (e.g., "classification", "regression").
        metric (Optional[MetricName]): The user-specified metric, or None to use default.

    Returns:
        MetricName: A valid metric name compatible with the given task.

    Raises:
        ValueError: If the metric is invalid for the given task.
        ValueError: If the task type is unknown.
    """

    # ───────── Default Metric Resolution ─────────
    if metric is None:
        metric = DEFAULT_METRIC[task]

    # ───────── Classification Validation ─────────
    if task == "classification":
        if metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"Invalid classification metric '{metric}'. "
                f"Allowed: {list(CLASSIFICATION_METRICS.keys())}"
            )
        return metric

    # ───────── Regression Validation ─────────
    if task == "regression":
        if metric not in REGRESSION_METRICS:
            raise ValueError(
                f"Invalid regression metric '{metric}'. "
                f"Allowed: {list(REGRESSION_METRICS.keys())}"
            )
        return metric

    # ───────── Unknown Task ─────────
    raise ValueError(f"Unknown task: {task}")
