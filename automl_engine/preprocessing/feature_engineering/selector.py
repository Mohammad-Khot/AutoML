from automl_engine.preprocessing.strategies.selectors import get_selector


def build_selector(config):
    """Build feature selection from the resolved strategy and dataset metadata."""
    return get_selector(
        task=config.problem.task,
        mode=config.preprocessing.feature_selection_method,
        n_features=config.artifacts.data_info.n_features,
    )
