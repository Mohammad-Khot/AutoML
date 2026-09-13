from typing import Optional, List

from automl_engine.planning.metadata import DataInfo
from automl_engine.planning.models.spec import ModelSpec

MAX_ROWS: int = 200_000
MAX_FEATURES: int = 5_000


def is_model_suitable(
    spec: ModelSpec,
    info: DataInfo,
    *,
    allow_missing_via_pipeline: bool = False,
) -> bool:
    """Return whether a model is compatible with the dataset and pipeline."""
    return _unsuitable_reason(
        spec,
        info,
        allow_missing_via_pipeline=allow_missing_via_pipeline,
    ) is None


def _unsuitable_reason(
    spec: ModelSpec,
    info: DataInfo,
    *,
    allow_missing_via_pipeline: bool = False,
) -> Optional[List[str]]:
    reasons: List[str] = []

    if spec.sensitive_to_dataset_size and info.n_rows > MAX_ROWS:
        reasons.append("too many rows for size-sensitive model.")

    if info.n_rows > MAX_ROWS and not spec.scales_to_large_datasets:
        reasons.append("not suitable for large datasets.")

    # ModelSpec currently has no handles_high_dim field. Use getattr so a
    # wide dataset does not crash the resolver with AttributeError.
    if info.n_features > MAX_FEATURES and not getattr(spec, "handles_high_dim", False):
        reasons.append("too high dimensional.")

    # Missing values are safe for non-native models when the preprocessing
    # pipeline is configured to impute them before model fitting.
    if (
        info.has_missing
        and not allow_missing_via_pipeline
        and not spec.handles_missing_values
    ):
        reasons.append("does not support missing values.")

    if info.is_sparse and not spec.supports_sparse_input:
        reasons.append("does not support sparse input.")

    return reasons if reasons else None
