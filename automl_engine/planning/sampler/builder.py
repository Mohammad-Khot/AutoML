from math import ceil

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from automl_engine.planning.experiment.resolved import ResolvedConfig


def build_sampler(resolved: ResolvedConfig):
    method = resolved.sampling.method

    if method in ("none", None):
        return None

    seed = resolved.runtime.seed
    strategy = resolved.sampling.strategy

    if method in ("smote", "adasyn"):
        info = resolved.artifacts.data_info
        minority_count = 0
        if info.minority_ratio is not None:
            minority_count = max(1, int(round(info.minority_ratio * info.n_rows)))

        # Approximate the smallest minority count available in a CV training
        # fold. Synthetic samplers need at least two observations there.
        folds = max(2, resolved.cv.folds)
        train_minority = minority_count - ceil(minority_count / folds)

        if train_minority < 2:
            return RandomUnderSampler(
                sampling_strategy=strategy,
                random_state=seed,
            )

        neighbors = min(resolved.sampling.k_neighbors, train_minority - 1)

        if method == "smote":
            return SMOTE(
                sampling_strategy=strategy,
                k_neighbors=neighbors,
                random_state=seed,
            )

        return ADASYN(
            sampling_strategy=strategy,
            n_neighbors=neighbors,
            random_state=seed,
        )

    if method == "undersample":
        return RandomUnderSampler(
            sampling_strategy=strategy,
            random_state=seed,
        )

    if method == "auto":
        raise ValueError("Sampling 'auto' must be resolved before building.")

    raise ValueError(f"Unknown sampling method: {method}")
