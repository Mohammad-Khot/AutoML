from sklearn.pipeline import Pipeline

from ..feature_engineering.vector import VectorFE
from ..feature_engineering.selector import build_selector
from ..feature_engineering.dimensionality_reduction import (
    build_dimensionality_reduction,
)
from ...planning.experiment.resolved import ResolvedConfig
from ...planning.models import ModelSpec


def build_global_pipeline(resolved: ResolvedConfig, model_spec: ModelSpec):
    steps = []

    # Only apply vector FE if enabled
    if resolved.feature_generation.method != "none":
        steps.append(("fe_vector", VectorFE(resolved, model_spec=model_spec)))

    selector = build_selector(resolved)
    if selector != "passthrough":
        steps.append(("selector", selector))

    dr = build_dimensionality_reduction(resolved)
    if dr != "passthrough":
        steps.append(("dr", dr))

    return Pipeline(steps)
