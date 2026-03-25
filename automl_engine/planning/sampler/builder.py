from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from automl_engine.planning.experiment.resolved import ResolvedConfig


def build_sampler(resolved: ResolvedConfig):
    method = resolved.sampling.method

    if method in ("none", None):
        return None

    if method == "smote":
        return SMOTE(
            sampling_strategy=resolved.sampling.strategy,
            k_neighbors=resolved.sampling.k_neighbors
        )

    if method == "adasyn":
        return ADASYN(
            sampling_strategy=resolved.sampling.strategy,
            n_neighbors=resolved.sampling.k_neighbors
        )

    if method == "undersample":
        return RandomUnderSampler(
            sampling_strategy=resolved.sampling.strategy
        )

    if method == "auto":
        raise ValueError("Sampling 'auto' must be resolved before building.")

    raise ValueError(f"Unknown sampling method: {method}")
