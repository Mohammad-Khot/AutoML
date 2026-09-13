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

_METRIC_TO_SKLEARN = {**CLASSIFICATION_METRICS, **REGRESSION_METRICS}


def get_scorer_safe(metric_name: str) -> Callable:
    """Resolve AutoML metric aliases to sklearn scorer objects."""
    sklearn_name = _METRIC_TO_SKLEARN.get(metric_name, metric_name)
    try:
        return get_scorer(sklearn_name)
    except ValueError as exc:
        raise ValueError(f"Unknown sklearn scorer: {metric_name}") from exc


def get_scorer_name(metric_name: str) -> str:
    """Return the sklearn scoring-string equivalent for an AutoML metric."""
    return _METRIC_TO_SKLEARN.get(metric_name, metric_name)


def resolve_metric(task: MLTask, metric: Optional[MetricName]) -> MetricName:
    if metric is None:
        metric = DEFAULT_METRIC[task]

    if task == "classification":
        if metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"Invalid classification metric '{metric}'. "
                f"Allowed: {list(CLASSIFICATION_METRICS.keys())}"
            )
        return metric

    if task == "regression":
        if metric not in REGRESSION_METRICS:
            raise ValueError(
                f"Invalid regression metric '{metric}'. "
                f"Allowed: {list(REGRESSION_METRICS.keys())}"
            )
        return metric

    raise ValueError(f"Unknown task: {task}")
