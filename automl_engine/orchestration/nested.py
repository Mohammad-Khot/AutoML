from automl_engine.preprocessing import build_pipeline
from automl_engine.core import select_best_model, MODEL_PRIORITY
from automl_engine.evaluation import get_scorer_safe
from automl_engine.evaluation.cv import get_cv_object
from automl_engine.reporting import print_subsection, log_model_score


def nested_cv(X, y, models, outer_cv, config, resolved):
    """
    Perform nested cross-validation.

    Parameters
    ----------
    X, y : dataset
    models : candidate models
    outer_cv : outer CV splitter
    resolved : resolved experiment configuration
    config : runtime/search policy
    """

    if getattr(outer_cv, "n_splits", 0) < 2:
        raise ValueError(
            "Outer CV has fewer than 2 splits. "
            "Dataset too small for nested CV"
        )

    outer_scores = []
    selected_models = []

    scorer = get_scorer_safe(resolved.metric)
    task = resolved.task

    from automl_engine.evaluation import evaluate_models

    # ======================================================
    # OUTER LOOP
    # ======================================================
    for i, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y), 1
    ):
        if config.log:
            print_subsection(f"Outer Fold {i}/{outer_cv.n_splits}")

        X_train = X.iloc[outer_train_idx]
        X_test = X.iloc[outer_test_idx]
        y_train = y.iloc[outer_train_idx]
        y_test = y.iloc[outer_test_idx]

        # --------------------------------------------------
        # Degenerate classification fold safeguard
        # --------------------------------------------------
        if task == "classification" and y_train.value_counts().min() < 2:
            print("[WARN] Fold has classes with <2 samples. Using dummy model.")

            best_name = (
                "dummy" if "dummy" in models else next(iter(models))
            )

            best_info = models[best_name]

            pipeline = build_pipeline(
                best_info,
                X_train,
                config,
                seed=config.seed,
            )

            pipeline.fit(X_train, y_train)

            score = scorer(pipeline, X_test, y_test)

            outer_scores.append(score)
            selected_models.append(best_name)
            continue

        # --------------------------------------------------
        # INNER CV
        # --------------------------------------------------
        safe_inner_cv = get_cv_object(
            task,
            y_train,
            max(2, config.cv_folds - 1),
            config.seed,
        )

        state = evaluate_models(
            X_train,
            y_train,
            models,
            safe_inner_cv,
            config,
            resolved,
            f"INNER-{i}"
        )

        if not state.scores:
            raise RuntimeError(
                "All models failed during inner CV evaluation."
            )

        # --------------------------------------------------
        # Select best inner model
        # --------------------------------------------------
        best_name = select_best_model(
            state.scores,
            MODEL_PRIORITY,
        )

        selected_models.append(best_name)
        best_info = models[best_name]

        # --------------------------------------------------
        # Refit on full outer train
        # --------------------------------------------------
        pipeline = build_pipeline(
            best_info,
            X_train,
            config,
            seed=config.seed,
        )

        pipeline.fit(X_train, y_train)

        score = scorer(pipeline, X_test, y_test)
        log_model_score(
            best_name,
            score,
            stage=f"OUTER-{i}",
            log=config.log
        )
        outer_scores.append(score)

    return outer_scores
