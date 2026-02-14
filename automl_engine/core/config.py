# core/config.py

from dataclasses import dataclass
from typing import Optional, Literal, get_args, Any, Type, Self

METRIC_TASK_MAP = {
    "accuracy": "classification",
    "f1": "classification",
    "roc_auc": "classification",

    "rmse": "regression",
    "mae": "regression",
    "r2": "regression",
}

CLASSIFICATION_METRICS = {
    metric for metric, task in METRIC_TASK_MAP.items() if task == "classification"
}

REGRESSION_METRICS = {
    metric for metric, task in METRIC_TASK_MAP.items() if task == "regression"
}

ScalingMode = Literal["auto", "force", "none"]
ScalerType = Literal["standard", "minmax", "robust", "maxabs"]
EncodingMode = Literal["auto", "onehot", "ordinal", "none"]
TaskType = Literal["classification", "regression"]
FeatureSelection = Literal["auto", "none", "kbest", "variance"]
MetricType = Literal["accuracy", "f1", "roc_auc", "rmse", "mae", "r2"]
MaxCompute = Literal["low", "medium", "high"]
LeakPolicy = Literal["error", "warn", "drop"]
CVStrategy = Literal["auto", "kfold", "stratified", "repeated", "timeseries"]
ImputeStrategy = Literal["auto", "simple", "knn", "iterative", "none"]


@dataclass
class AutoMLConfig:
    """Central configuration for AutoML pipeline."""

    # ─────────────── Core Settings ───────────────
    metric: Optional[MetricType] = None
    target: Optional[str] = None
    task: Optional[TaskType] = None

    cv_folds: int = 5
    nested_cv: bool = True
    cv_strategy: CVStrategy = "auto"
    seed: Optional[int] = None
    n_jobs: int = -1

    # ─────────────── Preprocessing ───────────────
    scaling_mode: ScalingMode = "auto"
    scaler_type: ScalerType = "standard"

    encoding_mode: EncodingMode = "auto"
    max_cardinality_one_hot: int = 20

    feature_selection: FeatureSelection = "auto"

    imputation: ImputeStrategy = "auto"
    add_missing_indicator: bool = False

    # ─────────────── Model Control ───────────────
    max_compute: MaxCompute = "high"

    allowed_models: Optional[list[str]] = None
    blocked_models: Optional[list[str]] = None  # Planned feature
    time_budget_soft: int = 600  # advisory budget used to plan search aggressiveness,NOT a hard termination limit

    min_improvement_over_dummy: float = 0.01

    # ─────────────── Data Quality ───────────────
    drop_leaky_columns: LeakPolicy = None  # Add logging
    id_threshold: float = 0.95

    # ─────────────── Search Strategy ───────────────
    scout_fraction: float = 0.2
    scout_folds: int = 3

    # ─────────────── Misc ───────────────
    log: bool = True

    @classmethod
    def fast(cls, **overrides: Any) -> Self:
        return cls(
            cv_folds=3,
            nested_cv=False,
            max_compute="low",
            scout_fraction=0.1,
            scout_folds=2,
            time_budget_soft=120,
            **overrides
        )

    @classmethod
    def thorough(cls, **overrides: Any) -> Self:
        return cls(
            cv_folds=10,
            nested_cv=True,
            max_compute="high",
            scout_fraction=0.4,
            scout_folds=5,
            time_budget_soft=1800,
            **overrides
        )

    @staticmethod
    def _validate_literal(value: Any, literal_type: Type, name: str) -> None:
        args = get_args(literal_type)

        if not args:
            raise TypeError(f"{name} is not a literal type.")

        if value not in args:
            allowed = ", ".join(map(str, args))
            raise ValueError(f"invalid {name}: {value}. Allowed: {allowed}")

    def __post_init__(self):
        self._validate_literal(self.scaling_mode, ScalingMode, "scaling_mode")
        self._validate_literal(self.scaler_type, ScalerType, "scaler_type")
        self._validate_literal(self.encoding_mode, EncodingMode, "encoding_mode")

        if self.task is not None:
            self._validate_literal(self.task, TaskType, "task_type")

        if self.metric is not None:
            self._validate_literal(self.metric, MetricType, "metric_type")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")

        if not (0 < self.scout_fraction <= 1):
            raise ValueError("scout_fraction must be in (0, 1]")

        if self.task and self.metric:
            expected = METRIC_TASK_MAP[self.metric]
            if expected != self.task:
                raise ValueError(
                    f"Metric '{self.metric}' belongs to {expected}, "
                    f"not {self.task}"
                )

        if self.time_budget_soft <= 0:
            raise ValueError("time_budget must be positive")

        if self.nested_cv and self.scout_folds >= self.cv_folds:
            raise ValueError("scout_folds must be smaller than cv_folds")

    def __repr__(self):
        return f"AutoMLConfig(task={self.task}, metric={self.metric}, cv={self.cv_folds})"
