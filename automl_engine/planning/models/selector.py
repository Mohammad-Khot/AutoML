from typing import Optional

from automl_engine.planning.metadata import DataInfo
from automl_engine.planning.models.spec import ModelSpec

MAX_ROWS: int = 200_000
MAX_FEATURES: int = 5_000


def is_model_suitable(name: str, spec: ModelSpec, info: DataInfo) -> bool:
    """
    Determine whether a model is suitable for the given dataset metadata.
    """
    return _unsuitable_reason(name, spec, info) is None


def _unsuitable_reason(
    name: str,
    spec: ModelSpec,
    info: DataInfo,
) -> Optional[str]:
    """
    Return the reason why a model is unsuitable for the given dataset.
    """

    # -------------------------
    # Large dataset checks
    # -------------------------

    if spec.size_sensitive and info.n_rows > MAX_ROWS:
        return f"{name}: too many rows for size-sensitive model."

    if info.n_rows > MAX_ROWS and not spec.handles_large_datasets:
        return f"{name}: not suitable for large datasets."

    # -------------------------
    # High dimensional checks
    # -------------------------

    if info.n_features > MAX_FEATURES and not spec.handles_high_dim:
        return f"{name}: too high dimensional."

    # -------------------------
    # Missing value handling
    # -------------------------

    if info.has_missing and not spec.supports_missing:
        return f"{name}: does not support missing values."

    # -------------------------
    # Sparse matrix support
    # -------------------------

    if info.is_sparse and not spec.supports_sparse:
        return f"{name}: does not support sparse input."

    # -------------------------
    # Small dataset heuristics
    # -------------------------

    if info.n_rows < 2000 and not spec.recommended_for_small_data:
        # not a hard rejection — just allow it
        pass

    return None
