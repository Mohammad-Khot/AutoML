# core/selector.py

from automl_engine.planning.metadata import DataInfo

MAX_ROWS = 200_000
MAX_FEATURES = 5000


def is_model_suitable(name: str, meta: dict, info: DataInfo) -> bool:
    return not _unsuitable_reason(name, meta, info)


def _unsuitable_reason(name: str, meta: dict, info: DataInfo) -> str | None:
    if meta.get("size_sensitive", False) and info.n_rows > MAX_ROWS:
        return f"{name}: too many rows for size-sensitive model."

    if not meta.get("handles_high_dim", False) and info.n_features > MAX_FEATURES:
        return f"{name}: too high dimensional."

    if info.has_categorical and not meta.get("native_categorical", False):
        return f"{name}: lacks categorical support."

    return None
