from typing import Optional, List

from automl_engine.planning.metadata import DataInfo
from automl_engine.planning.models.spec import ModelSpec

MAX_ROWS: int = 200_000
MAX_FEATURES: int = 5_000


def is_model_suitable(spec: ModelSpec, info: DataInfo) -> bool:
    """
    Check if a model is suitable for a given dataset.

    This function acts as a wrapper over `_unsuitable_reason` and
    returns True if no constraints are violated.

    Args:
        spec (ModelSpec): Model capability specification.
        info (DataInfo): Metadata describing the dataset.

    Returns:
        bool: True if the model is suitable, False otherwise.

    Raises:
        None
    """
    return _unsuitable_reason(spec, info) is None


def _unsuitable_reason(
    spec: ModelSpec,
    info: DataInfo,
) -> Optional[List[str]]:
    """
    Identify why a model is unsuitable for a dataset.

    Evaluates dataset size, dimensionality, missing values,
    sparsity, and model capability constraints. Returns the
    first encountered incompatibility reason.

    Args:
        spec (ModelSpec): Model capability specification.
        info (DataInfo): Metadata describing the dataset.

    Returns:
        Optional[List[str]]: Reason for unsuitability, or None if suitable.

    Raises:
        None
    """

    reasons: List[str] = []

    # -------------------------
    # Large dataset checks
    # -------------------------

    if spec.sensitive_to_dataset_size and info.n_rows > MAX_ROWS:
        reasons.append("too many rows for size-sensitive model.")

    if info.n_rows > MAX_ROWS and not spec.scales_to_large_datasets:
        reasons.append("not suitable for large datasets.")

    # -------------------------
    # High dimensional checks
    # -------------------------

    if info.n_features > MAX_FEATURES and not spec.handles_high_dim:
        return [f"{spec.name}: too high dimensional."]

    # -------------------------
    # Missing value handling
    # -------------------------

    if info.has_missing and not spec.handles_missing_values:
        return [f"{spec.name}: does not support missing values."]

    # -------------------------
    # Sparse matrix support
    # -------------------------

    if info.is_sparse and not spec.supports_sparse_input:
        return [f"{spec.name}: does not support sparse input."]

    # -------------------------
    # Small dataset heuristics
    # -------------------------

    if info.n_rows < 2000 and not spec.suitable_for_small_datasets:
        # not a hard rejection — just allow it
        pass

    return reasons if reasons else None
