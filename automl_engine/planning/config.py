# planning/config.py

from dataclasses import dataclass, field
from typing import Optional, Literal, get_args, Any, Type, Self

# ─────────────── Literal Types ───────────────

ScalingMode = Literal["auto", "force", "none"]
ScalerKind = Literal["auto", "standard", "minmax", "robust", "maxabs"]

EncodingStrategy = Literal["auto", "onehot", "ordinal", "none"]

MLTask = Literal["classification", "regression"]

FeatureSelectionMethod = Literal["auto", "none", "variance", "l1", "tree"]

MetricName = Literal[
    "accuracy", "f1", "f1_macro", "roc_auc", "precision", "recall",
    "mse", "rmse", "mae", "r2"
]

ComputeBudget = Literal["low", "medium", "high"]

LeakHandlingPolicy = Literal["error", "warn", "drop"]

CrossValidationStrategy = Literal["auto", "kfold", "stratified", "repeated", "timeseries"]

ImputationStrategy = Literal["auto", "simple", "knn", "iterative", "none"]

FeatureGenerationMethod = Literal[
    "auto",
    "none",
    "polynomial",
    "interactions",
    "datetime",
    "text",
    "hashing",
]

FeatureGenerationStrategy = Literal[
    "safe",
    "aggressive",
    "minimal"
]

DimensionalityReductionMethod = Literal[
    "auto",
    "none",
    "pca",
    "svd",
    "ica",
]

SamplingMethod = Literal["auto", "none", "smote", "adasyn", "undersample"]
SamplingStrategy = Literal["auto"] | float


# ─────────────── Problem Definition ───────────────

@dataclass
class ProblemConfig:
    target: Optional[str] = None
    task: Optional[MLTask] = None
    metric: Optional[MetricName] = None


# ─────────────── Cross Validation ───────────────

@dataclass
class CVConfig:
    folds: int = 5
    strategy: CrossValidationStrategy = "auto"
    use_nested_cv: bool = False
    repeats: int = 1


# ─────────────── Preprocessing ───────────────

@dataclass
class PreprocessingConfig:
    scaling_mode: ScalingMode = "auto"
    scaler_kind: ScalerKind = "auto"

    encoding_strategy: EncodingStrategy = "auto"
    max_cardinality_one_hot: int = 20

    feature_selection_method: FeatureSelectionMethod = "auto"

    imputation_strategy: ImputationStrategy = "auto"
    add_missing_indicator: bool = False


# ─────────────── Feature Generation ───────────────

@dataclass
class FeatureGenerationConfig:
    method: FeatureGenerationMethod = "auto"
    strategy: FeatureGenerationStrategy = "safe"
    max_polynomial_degree: int = 2
    interaction_only: bool = False
    max_generated_features: int = 100
    subsample_ratio: float = 1.0


# ─────────────── Dimensionality Reduction ───────────────

@dataclass
class DimensionalityReductionConfig:
    method: DimensionalityReductionMethod = "auto"
    n_components: int = 0.95
    variance_threshold: float = 0.95
    apply_after_generation: bool = True


# ─────────────── Model Space ───────────────

@dataclass
class ModelSpaceConfig:
    include_models: Optional[list[str]] = None
    exclude_models: Optional[list[str]] = None
    top_k_models: int = 4


# ─────────────── Search Strategy ───────────────

@dataclass
class SearchConfig:
    compute_budget: ComputeBudget = "high"

    scout_sample_fraction: float = 0.2
    scout_folds: int = 3

    time_budget_soft: int = 600
    min_improvement_over_dummy: float = 0.01


# ─────────────── Data Quality ───────────────

@dataclass
class DataQualityConfig:
    leak_handling: LeakHandlingPolicy = "warn"
    id_threshold: float = 0.95


# ─────────────── Runtime Settings ───────────────

@dataclass
class RuntimeConfig:
    seed: int = 42
    n_jobs: int = -1
    log: bool = True


# ─────────────── Optuna Optimization ───────────────

@dataclass
class OptunaConfig:
    enabled: bool = True
    n_trials: int = 50
    direction: str = "maximize"
    n_jobs: int = 1
    seed: int | None = None


# ─────────────── Sampling Method ───────────────


@dataclass
class SamplingConfig:
    method: SamplingMethod = "auto"
    strategy: SamplingStrategy = "auto"
    k_neighbors: int = 5


# ─────────────── Root AutoML Config ───────────────

@dataclass
class AutoMLConfig:
    """
    Central configuration object controlling the AutoML pipeline.
    """

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    feature_generation: FeatureGenerationConfig = field(default_factory=FeatureGenerationConfig)
    dimensionality_reduction: DimensionalityReductionConfig = field(default_factory=DimensionalityReductionConfig)

    models: ModelSpaceConfig = field(default_factory=ModelSpaceConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    optuna: OptunaConfig = field(default_factory=OptunaConfig)

    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    generate_optuna_plots: bool = True
    display_optuna_plots: bool = True

    # ─────────────── Constructors ───────────────

    @classmethod
    def fast(cls, **overrides: Any) -> Self:
        config = cls(**overrides)

        config.cv.folds = 3
        config.cv.use_nested_cv = False

        config.search.compute_budget = "low"
        config.search.scout_sample_fraction = 0.1
        config.search.scout_cv_folds = 2
        config.search.time_budget_soft = 120

        return config

    @classmethod
    def thorough(cls, **overrides: Any) -> Self:
        config = cls(**overrides)

        config.cv.folds = 10
        config.cv.use_nested_cv = True

        config.search.compute_budget = "high"
        config.search.scout_sample_fraction = 0.4
        config.search.scout_cv_folds = 5
        config.search.time_budget_soft = 1800

        return config

    # ─────────────── Validation Helpers ───────────────

    @staticmethod
    def _validate_literal(value: Any, literal_type: Type, name: str) -> None:
        args = get_args(literal_type)

        if not args:
            raise TypeError(f"{name} is not a literal type.")

        if value not in args:
            allowed = ", ".join(map(str, args))
            raise ValueError(f"invalid {name}: {value}. Allowed: {allowed}")

    # ─────────────── Post Init Validation ───────────────

    def __post_init__(self) -> None:

        p = self.preprocessing

        self._validate_literal(p.scaling_mode, ScalingMode, "scaling_mode")
        self._validate_literal(p.scaler_kind, ScalerKind, "scaler_kind")
        self._validate_literal(p.encoding_strategy, EncodingStrategy, "encoding_strategy")

        self._validate_literal(
            p.feature_selection_method,
            FeatureSelectionMethod,
            "feature_selection_method"
        )

        self._validate_literal(
            p.imputation_strategy,
            ImputationStrategy,
            "imputation_strategy"
        )

        # Feature generation
        fg = self.feature_generation
        self._validate_literal(fg.method, FeatureGenerationMethod, "feature_generation_method")

        if fg.max_polynomial_degree < 1:
            raise ValueError("max_polynomial_degree must be >= 1")

        if fg.max_generated_features <= 0:
            raise ValueError("max_generated_features must be > 0")

        if not (0 < fg.subsample_ratio <= 1):
            raise ValueError("subsample_ratio must be in (0,1]")

        # Dimensionality reduction
        dr = self.dimensionality_reduction
        self._validate_literal(dr.method, DimensionalityReductionMethod, "dimensionality_reduction_method")

        if dr.n_components is not None and dr.n_components <= 0:
            raise ValueError("n_components must be > 0")

        if not (0 < dr.variance_threshold <= 1):
            raise ValueError("variance_threshold must be in (0,1]")

        # Problem
        prob = self.problem

        if prob.task is not None:
            self._validate_literal(prob.task, MLTask, "task")

        if prob.metric is not None:
            self._validate_literal(prob.metric, MetricName, "metric")

        # CV
        cv = self.cv
        search = self.search

        if cv.folds < 2:
            raise ValueError("cv.folds must be at least 2")

        if cv.use_nested_cv and search.scout_folds >= cv.folds:
            raise ValueError("scout_folds must be smaller than folds")

        if cv.repeats < 1:
            raise ValueError("CV repeats must be >= 1.")

        if cv.strategy != "repeated" and cv.repeats > 1:
            raise ValueError("CV repeats > 1 is ignored unless strategy='repeated'.")

        # Search
        s = self.search

        if not (0 < s.scout_sample_fraction <= 1):
            raise ValueError("scout_sample_fraction must be in (0, 1]")

        if s.time_budget_soft <= 0:
            raise ValueError("time_budget_soft must be positive")

        # --- Cross-field constraints (STRICT MODE) ---

        if self.problem.task == "regression" and self.cv.strategy == "stratified":
            raise ValueError(
                "Invalid config: 'stratified' CV cannot be used for regression."
            )

    def __repr__(self) -> str:
        return (
            f"AutoMLConfig("
            f"task={self.problem.task}, "
            f"metric={self.problem.metric}, "
            f"cv={self.cv.folds})"
        )
