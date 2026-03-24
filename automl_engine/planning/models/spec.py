from dataclasses import dataclass
from typing import Callable, Optional, Literal
from sklearn.base import BaseEstimator

ModelFamily = Literal["linear", "tree", "kernel", "distance", "neural", "baseline"]
FEBenefit = Literal["high", "medium", "low"]
ComputeCost = Literal["low", "medium", "high"]
TuningSensitivity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], BaseEstimator]

    # core identity
    family: ModelFamily

    # preprocessing / behavior
    requires_scaling: bool
    sensitive_to_dataset_size: bool
    # handles_high_dim: bool = False
    # native_categorical: bool = False
    # interpretable: bool = False

    # feature engineering intelligence
    captures_feature_interactions: bool
    feature_engineering_impact: FEBenefit
    prefers_log_transformed_features: bool
    supports_nonlinearity: bool

    # optimization intelligence
    tuning_complexity: TuningSensitivity

    # compute characteristics
    training_cost: ComputeCost
    # supports_gpu: bool = False

    # data compatibility
    supports_sparse_input: bool
    handles_missing_values: bool

    # dataset heuristics
    scales_to_large_datasets: bool
    suitable_for_small_datasets: bool

    # tuning
    hyperparameter_space: Optional[Callable]

    # scheduling
    tie_breaker_score: int
