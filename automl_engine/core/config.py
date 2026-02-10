from dataclasses import dataclass
from typing import Optional, List, Literal, get_args, cast

ScalingMode = Literal["auto", "force", "none"]
ScalerType = Literal["standard", "minmax", "robust", "maxabs"]
EncodingMode = Literal["auto", "onehot", "ordinal", "none"]
TaskType = Literal["classification", "regression"]
FeatureSelection = Literal["auto", "none", "kbest", "variance"]
MetricType = Literal["accuracy", "f1", "roc_auc", "rmse", "mae", "r2"]


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
    allowed_models: Optional[List[str]] = None
    time_budget: int = 600  # seconds (soft limit)

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
        return cls(cv_folds=3, nested_cv=False, time_budget=120)

    @classmethod
    def thorough(cls):
        return cls(cv_folds=10, nested_cv=True, time_budget=1800)

    def __post_init__(self):
        if self.scaling_mode not in get_args(ScalingMode):
            raise ValueError(f"invalid scaling_mode : {self.scaling_mode}")

        if self.scaler_type not in get_args(ScalerType):
            raise ValueError(f"invalid scaler_type : {self.scaler_type}")

        if self.encoding_mode not in get_args(EncodingMode):
            raise ValueError(f"invalid encoding_mode : {self.encoding_mode}")

        if self.task is not None and self.task not in get_args(TaskType):
            raise ValueError(f"invalid task : {self.task}")

        if self.metric is not None and self.metric not in get_args(MetricType):
            raise ValueError(f"invalid metric : {self.metric}")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")

        if not (0 < self.scout_fraction <= 1):
            raise ValueError("scout_fraction must be in (0, 1]")

        if self.task == "regression" and self.metric in {"accuracy", "f1", "roc_auc"}:
            raise ValueError("Classification metrics cannot be used for regression")

        if self.task == "classification" and self.metric in {"rmse", "mae", "r2"}:
            raise ValueError("Regression metrics cannot be used for classification")

        if self.time_budget <= 0:
            raise ValueError("time_budget must be positive")

        if self.scout_folds >= self.cv_folds:
            raise ValueError("scout_folds must be smaller than cv_folds")

    def resolved_metric(self) -> MetricType | None:
        """Return concrete metric after task is known."""
        if self.task == "classification":
            return self.metric or cast(MetricType, "accuracy")
        if self.task == "regression":
            return self.metric or cast(MetricType, "r2")

        raise ValueError("Task must be set before selecting metric")

    def __repr__(self):
        return f"AutoMLConfig(task={self.task}, metric={self.metric}, cv={self.cv_folds})"
