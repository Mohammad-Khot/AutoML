from automl_engine.preprocessing import build_pipeline


def finalize_model(
    best_model_name,
    state,
    models,
    X,
    y,
    config,
    seed,
):

    best_info = models[best_model_name]

    pipeline = state.get_pipeline(best_model_name)

    if pipeline is None:
        pipeline = build_pipeline(
            best_info,
            X,
            config,
            seed=seed,
        )

    pipeline.fit(X, y)

    return pipeline
