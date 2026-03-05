from dataclasses import dataclass
from typing import Callable, Optional
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], BaseEstimator]

    # preprocessing / behavior
    needs_scaling: bool = False
    size_sensitive: bool = False
    handles_high_dim: bool = False
    native_categorical: bool = False
    interpretable: bool = False

    # compute characteristics
    compute_cost: str = "medium"
    supports_gpu: bool = False

    # data compatibility
    supports_sparse: bool = False
    supports_missing: bool = False

    # dataset heuristics
    handles_large_datasets: bool = False
    recommended_for_small_data: bool = False

    # tuning
    search_space: Optional[Callable] = None

    # scheduling
    priority: int = 2
