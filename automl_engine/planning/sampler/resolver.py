from automl_engine.planning.config import SamplingConfig, AutoMLConfig
from automl_engine.planning.metadata import DataInfo


def resolve_sampling(config: AutoMLConfig, data_info: DataInfo, task: str):

    # Only applies to classification
    if task != "classification":
        return SamplingConfig(method="none")

    # If user explicitly set something → respect it
    if config.sampling.method != "auto":
        return config.sampling

    ratio = data_info.minority_ratio

    if ratio is None:
        return SamplingConfig(method="none")

    # No imbalance
    if ratio > 0.4:
        return SamplingConfig(method="none")

    # Severe imbalance → SMOTE
    if ratio < 0.1:
        return SamplingConfig(method="smote")

    # Moderate imbalance → ADASYN
    if ratio < 0.2:
        return SamplingConfig(method="adasyn")

    # Mild imbalance → under sampling
    return SamplingConfig(method="undersample")
