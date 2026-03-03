from automl_engine.evaluation import evaluate_models
from automl_engine.orchestration.nested import nested_cv
from automl_engine.reporting import print_section


def execute_training_workflow(
    X,
    y,
    models,
    outer_cv,
    config,
    resolved,
):

    if not config.nested_cv:

        if config.log:
            print_section("Standard Cross Validation")

        state = evaluate_models(
            X, y,
            models,
            outer_cv,
            config,
            resolved,
            "OUTER_CV",
        )

        return state, state.scores

    # ---------- Nested ----------
    if config.log:
        print_section("Nested Evaluation")

    outer_result = nested_cv(
        X, y,
        models,
        outer_cv,
        config,
        resolved,
    )

    outer_scores = getattr(
        outer_result,
        "scores",
        outer_result
    )

    if config.log:
        print_section("Final Model Fit")

    state = evaluate_models(
        X, y,
        models,
        outer_cv,
        config,
        resolved,
        "FINAL_FIT",
    )

    return state, outer_scores
