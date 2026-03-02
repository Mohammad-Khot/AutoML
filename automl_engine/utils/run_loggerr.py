from automl_engine.utils.console import CONSOLE_WIDTH


def print_run_header(
    *,
    task,
    metric,
    n_samples,
    n_features,
    models,
    seed,
    cv,
    search_type,
):
    line = "=" * CONSOLE_WIDTH

    print("\n" + line)
    print("AUTO ML RUN".center(CONSOLE_WIDTH))
    print(line)

    print(f"Task          : {task}")
    print(f"Metric        : {metric}")
    print(f"Samples       : {n_samples}")
    print(f"Features      : {n_features}")
    print(f"Models        : {len(models)}")
    print(f"Seed          : {seed}")

    # CV description
    if hasattr(cv, "n_splits"):
        print(f"CV Strategy   : {cv.__class__.__name__} "
              f"(folds={cv.n_splits})")

    print(f"Search        : {search_type}")

    print("=" * CONSOLE_WIDTH + "\n")
