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
    print("\n" + "=" * 48)
    print("               AUTO ML RUN")
    print("=" * 48)

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

    print("=" * 48 + "\n")