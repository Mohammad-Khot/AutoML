from automl_engine.planning.models import (
    MODEL_PRIORITY,
    select_best_model,
)


def resolve_best_model(state):

    if not state.scores:
        raise RuntimeError(
            "No models successfully evaluated."
        )

    return select_best_model(
        state.scores,
        MODEL_PRIORITY,
    )
