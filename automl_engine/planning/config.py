# planning/config.py

from dataclasses import dataclass
from typing import Optional, Literal, get_args, Any, Type, Self


METRIC_TASK_MAP: dict[str, str] = {
    "accuracy": "classification",
    "f1": "classification",
    "roc_auc": "classification",
    "rmse": "regression",
    "mae": "regression",
    "r2": "regression",
}

CLASSIFICATION_METRICS: set[str] = {
    metric for metric, task in METRIC_TASK_MAP.items() if task == "classification"
}

REGRESSION_METRICS: set[str] = {
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
SearchStrategy = Literal[
    "grid", "random", "half_grid", "half_randomized", "bayesian"
]


@dataclass
class AutoMLConfig:
    """
    Central configuration object for controlling the AutoML pipeline.

    This class validates consistency between task and metric, enforces
    logical parameter constraints (e.g., CV settings, time budget),
    and provides convenience constructors for fast and thorough modes.
    """

    # ─────────────── Core Settings ───────────────
    metric: Optional[MetricType] = None
    target: Optional[str] = None
    task: Optional[TaskType] = None

    cv_folds: int = 5
    nested_cv: bool = False
    cv_strategy: CVStrategy = "auto"
    seed: Optional[int] = None
    n_jobs: int = -1
    top_k_models: int = 4

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
    blocked_models: Optional[list[str]] = None
    time_budget_soft: int = 600

    min_improvement_over_dummy: float = 0.01

    # ─────────────── Data Quality ───────────────
    drop_leaky_columns: Optional[LeakPolicy] = None
    id_threshold: float = 0.95

    # ─────────────── Search Strategy ───────────────
    scout_fraction: float = 0.2
    scout_folds: int = 3
    search_type: Optional[SearchStrategy] = None

    # ─────────────── Misc ───────────────
    log: bool = True

    @classmethod
    def fast(cls, **overrides: Any) -> Self:
        """
        Create a lightweight configuration optimized for speed.

        Returns:
            AutoMLConfig: Configuration tuned for quick experimentation.
        """
        return cls(
            cv_folds=3,
            nested_cv=False,
            max_compute="low",
            scout_fraction=0.1,
            scout_folds=2,
            time_budget_soft=120,
            **overrides,
        )

    @classmethod
    def thorough(cls, **overrides: Any) -> Self:
        """
        Create a comprehensive configuration optimized for robustness.

        Returns:
            AutoMLConfig: Configuration tuned for exhaustive evaluation.
        """
        return cls(
            cv_folds=10,
            nested_cv=True,
            max_compute="high",
            scout_fraction=0.4,
            scout_folds=5,
            time_budget_soft=1800,
            **overrides,
        )

    @staticmethod
    def _validate_literal(value: Any, literal_type: Type, name: str) -> None:
        """
        Validate that a value belongs to a given Literal type.

        Args:
            value: The value to validate.
            literal_type: The Literal type to validate against.
            name: The parameter name for error reporting.

        Raises:
            TypeError: If the provided type is not a Literal.
            ValueError: If the value is not within the allowed Literal options.
        """
        args = get_args(literal_type)

        if not args:
            raise TypeError(f"{name} is not a literal type.")

        if value not in args:
            allowed = ", ".join(map(str, args))
            raise ValueError(f"invalid {name}: {value}. Allowed: {allowed}")

    def __post_init__(self) -> None:
        """
        Perform validation checks after initialization.

        Ensures consistency between metric and task, validates
        literal-typed fields, and enforces logical parameter constraints.
        """
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
                    f"Metric '{self.metric}' belongs to {expected}, not {self.task}"
                )

        if self.time_budget_soft <= 0:
            raise ValueError("time_budget_soft must be positive")

        if self.nested_cv and self.scout_folds >= self.cv_folds:
            raise ValueError("scout_folds must be smaller than cv_folds")

    def __repr__(self) -> str:
        """
        Return a concise string representation of the configuration.

        Returns:
            str: Simplified configuration summary.
        """
        return (
            f"AutoMLConfig(task={self.task}, "
            f"metric={self.metric}, "
            f"cv={self.cv_folds})"
        )