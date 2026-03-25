# planning/experiment/resolved.py

from dataclasses import dataclass
from typing import Optional, Any

from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelEncoder

from automl_engine.planning.config import (
    MetricName,
    MLTask,
    ScalingMode,
    ScalerKind,
    EncodingStrategy,
    FeatureSelectionMethod,
    ComputeBudget,
    LeakHandlingPolicy,
    CrossValidationStrategy,
    ImputationStrategy,
    FeatureGenerationMethod,
    DimensionalityReductionMethod, FeatureGenerationStrategy, SamplingMethod, SamplingStrategy,
)
from automl_engine.planning.metadata import DataInfo
from automl_engine.planning.models import ModelSpec


# ─────────────── Resolved Optuna ───────────────

@dataclass(slots=True)
class ResolvedOptunaConfig:
    enabled: bool
    n_trials: int
    direction: str
    n_jobs: int
    seed: Optional[int]


# ─────────────── Problem ───────────────

@dataclass(slots=True)
class ResolvedProblemConfig:
    task: MLTask
    metric: MetricName
    target: str


# ─────────────── Cross Validation ───────────────

@dataclass(slots=True)
class ResolvedCVConfig:
    folds: int
    strategy: CrossValidationStrategy
    use_nested_cv: bool
    repeats: int


# ─────────────── Preprocessing ───────────────

@dataclass(slots=True)
class ResolvedPreprocessingConfig:
    scaling_mode: ScalingMode
    scaler_kind: ScalerKind

    encoding_strategy: EncodingStrategy
    max_cardinality_one_hot: int

    feature_selection_method: FeatureSelectionMethod

    imputation_strategy: ImputationStrategy
    add_missing_indicator: bool


# ─────────────── Feature Generation ───────────────

@dataclass(slots=True)
class ResolvedFeatureGenerationConfig:
    method: FeatureGenerationMethod
    strategy: FeatureGenerationStrategy
    max_polynomial_degree: int
    interaction_only: bool
    max_generated_features: int
    subsample_ratio: float


# ─────────────── Dimensionality Reduction ───────────────

@dataclass(slots=True)
class ResolvedDimensionalityReductionConfig:
    method: DimensionalityReductionMethod
    n_components: Optional[int]
    variance_threshold: float
    apply_after_generation: bool


# ─────────────── Model Space ───────────────

@dataclass(slots=True)
class ResolvedModelSpaceConfig:
    include_models: Optional[list[str]]
    exclude_models: Optional[list[str]]
    top_k_models: int
    compute_budget: ComputeBudget


# ─────────────── Search Strategy ───────────────

@dataclass(slots=True)
class ResolvedSearchConfig:
    scout_sample_fraction: float
    scout_folds: int
    time_budget_soft: int
    min_improvement_over_dummy: float


# ─────────────── Data Quality ───────────────

@dataclass(slots=True)
class ResolvedDataQualityConfig:
    leak_handling: Optional[LeakHandlingPolicy]
    id_threshold: float


# ─────────────── Runtime ───────────────

@dataclass(slots=True)
class ResolvedRuntimeConfig:
    seed: Optional[int]
    n_jobs: int
    log: bool


# ─────────────── Artifacts ───────────────

@dataclass
class ResolvedArtifactsConfig:
    models: dict[str, ModelSpec]
    cv_object: BaseCrossValidator
    data_info: DataInfo
    label_encoder: LabelEncoder | None
    leaks: Any


# ─────────────── Sampling Method ───────────────


@dataclass
class SamplingConfig:
    method: SamplingMethod = "auto"
    strategy: SamplingStrategy = "auto"
    k_neighbors: int = 5


# ─────────────── Root Resolved Config ───────────────

@dataclass(slots=True)
class ResolvedConfig:
    problem: ResolvedProblemConfig

    cv: ResolvedCVConfig

    preprocessing: ResolvedPreprocessingConfig

    feature_generation: ResolvedFeatureGenerationConfig

    dimensionality_reduction: ResolvedDimensionalityReductionConfig

    models: ResolvedModelSpaceConfig

    search: ResolvedSearchConfig

    data_quality: ResolvedDataQualityConfig

    runtime: ResolvedRuntimeConfig

    optuna: ResolvedOptunaConfig

    artifacts: ResolvedArtifactsConfig

    sampling: SamplingConfig

    generate_optuna_plots: bool
    display_optuna_plots: bool
