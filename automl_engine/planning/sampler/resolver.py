from dataclasses import replace

from automl_engine.planning.config import SamplingConfig, AutoMLConfig
from automl_engine.planning.metadata import DataInfo


def resolve_sampling(
    config: AutoMLConfig,
    data_info: DataInfo,
    task: str,
) -> SamplingConfig:
    """Resolve an explicit sampling strategy from task/data metadata."""
    if task != "classification":
        return SamplingConfig(method="none")

    if (
        config.sampling.strategy != "auto"
        and data_info.n_classes is not None
        and data_info.n_classes != 2
    ):
        raise ValueError(
            "A float sampling.strategy is only supported for binary classification."
        )

    if config.sampling.method != "auto":
        # Never return the mutable user configuration object as resolved state.
        return replace(config.sampling)

    ratio = data_info.minority_ratio
    if ratio is None or ratio > 0.4:
        return SamplingConfig(method="none")

    if ratio < 0.1:
        return SamplingConfig(method="smote")

    if ratio < 0.2:
        return SamplingConfig(method="adasyn")

    return SamplingConfig(method="undersample")
