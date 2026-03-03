# planning/models/selector.py

from typing import Any, Dict, Optional

from automl_engine.planning.metadata import DataInfo

MAX_ROWS: int = 200_000
MAX_FEATURES: int = 5_000


def is_model_suitable(name: str, meta: Dict[str, Any], info: DataInfo) -> bool:
    """
    Determine whether a model is suitable for the given dataset metadata.

    A model is considered suitable if no exclusion rule applies based on:
    - Dataset size (row count)
    - Feature dimensionality
    - Categorical feature support

    Parameters
    ----------
    name : str
        The name of the model.
    meta : Dict[str, Any]
        Metadata describing model capabilities (e.g., size sensitivity,
        high-dimensional support, categorical handling).
    info : DataInfo
        Dataset metadata including row count, feature count, and categorical presence.

    Returns
    -------
    bool
        True if the model is suitable, False otherwise.
    """
    return _unsuitable_reason(name, meta, info) is None


def _unsuitable_reason(
    name: str,
    meta: Dict[str, Any],
    info: DataInfo,
) -> Optional[str]:
    """
    Return the reason why a model is unsuitable for the given dataset.

    The checks include:
    - Rejecting size-sensitive models when row count exceeds MAX_ROWS.
    - Rejecting models that do not handle high-dimensional data when
      feature count exceeds MAX_FEATURES.
    - Rejecting models that lack native categorical support when
      categorical features are present.

    Parameters
    ----------
    name : str
        The name of the model.
    meta : Dict[str, Any]
        Metadata describing model capabilities.
    info : DataInfo
        Dataset metadata including row count, feature count, and categorical presence.

    Returns
    -------
    Optional[str]
        A descriptive reason if the model is unsuitable, otherwise None.
    """
    if meta.get("size_sensitive", False) and info.n_rows > MAX_ROWS:
        return f"{name}: too many rows for size-sensitive model."

    if not meta.get("handles_high_dim", False) and info.n_features > MAX_FEATURES:
        return f"{name}: too high dimensional."

    if info.has_categorical and not meta.get("native_categorical", False):
        return f"{name}: lacks categorical support."

    return None
