# evaluation/metrics.py

from sklearn.metrics import get_scorer

CLASSIFICATION_METRICS = {
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
}

REGRESSION_METRICS = {
    "r2": "r2",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}

DEFAULT_METRIC = {
    "classification": "accuracy",
    "regression": "r2",
}


def get_scorer_safe(metric_name: str):
    try:
        scorer = get_scorer(metric_name)
    except ValueError:
        raise ValueError(f"Unknown metric : {metric_name}")
    return scorer


def resolve_metric(task: str, metric: str | None) -> str:
    if metric is None:
        return DEFAULT_METRIC[task]

    metric = metric.lower()

    if task == "classification":
        if metric not in CLASSIFICATION_METRICS:
            print(f"Metric {metric} invalid for classification.\n[METRIC] : Accuracy")
            return "accuracy"
        return CLASSIFICATION_METRICS[metric]

    if task == "regression":
        if metric not in REGRESSION_METRICS:
            print(f"Metric {metric} invalid for regression. \n[METRIC] : R2")
            return "r2"
        return REGRESSION_METRICS[metric]

    raise ValueError(f"Unknown task: {task}")
