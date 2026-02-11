# core/config.py

from dataclasses import dataclass
from typing import Optional, Literal, get_args

CLASSIFICATION_METRICS = {"accuracy", "f1", "roc_auc"}
REGRESSION_METRICS = {"rmse", "mae", "r2"}

ScalingMode = Literal["auto", "force", "none"]
ScalerType = Literal["standard", "minmax", "robust", "maxabs"]
EncodingMode = Literal["auto", "onehot", "ordinal", "none"]
TaskType = Literal["classification", "regression"]
FeatureSelection = Literal["auto", "none", "kbest", "variance"]
MetricType = Literal["accuracy", "f1", "roc_auc", "rmse", "mae", "r2"]
MaxCompute = Literal["low", "medium", "high"]


@dataclass
class AutoMLConfig:
    """Central configuration for AutoML pipeline."""

    # ─────────────── Core Settings ───────────────
    metric: Optional[MetricType] = None
    target: Optional[str] = None
    task: Optional[TaskType] = None

    cv_folds: int = 5
    nested_cv: bool = True
    seed: Optional[int] = None
    n_jobs: int = -1

    # ─────────────── Preprocessing ───────────────
    scaling_mode: ScalingMode = "auto"
    scaler_type: ScalerType = "standard"

    encoding_mode: EncodingMode = "auto"
    max_cardinality_one_hot: int = 20

    feature_selection: FeatureSelection = "auto"

    # ─────────────── Model Control ───────────────
    max_compute: MaxCompute = "high"

    allowed_models: Optional[list[str]] = None
    time_budget_soft: int = 600  # seconds (soft limit)

    min_improvement_over_dummy: float = 0.01  # min improvement over dummy

    # ─────────────── Data Quality ───────────────
    drop_leaky_columns: bool = False
    id_threshold: float = 0.95

    # ─────────────── Search Strategy ───────────────
    scout_fraction: float = 0.2
    scout_folds: int = 3

    # ─────────────── Misc ───────────────
    log: bool = True

    @classmethod
    def fast(cls):
        return cls(cv_folds=3, nested_cv=False, time_budget_soft=120)

    @classmethod
    def thorough(cls):
        return cls(cv_folds=10, nested_cv=True, time_budget_soft=1800)

    @staticmethod
    def _validate_literal(value, literal_type, name: str):
        if value not in get_args(literal_type):
            allowed = ", ".join(map(str, get_args(literal_type)))
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
            if self.task == "regression" and self.metric in CLASSIFICATION_METRICS:
                raise ValueError(f"{self.metric} is for classification, not regression")
            elif self.task == "classification" and self.metric in REGRESSION_METRICS:
                raise ValueError(f"{self.metric} is for regression, not classification")

        if self.time_budget_soft <= 0:
            raise ValueError("time_budget must be positive")

        if self.nested_cv and self.scout_folds >= self.cv_folds:
            raise ValueError("scout_folds must be smaller than cv_folds")

    def __repr__(self):
        return f"AutoMLConfig(task={self.task}, metric={self.metric}, cv={self.cv_folds})"
