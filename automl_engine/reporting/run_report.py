# automl_engine/reporting/run_report.py

from typing import Any, Final, Sequence

from .console import CONSOLE_WIDTH


LINE: Final[str] = "=" * CONSOLE_WIDTH


def print_run_header(
    *,
    task: str,
    metric: str,
    n_samples: int,
    n_features: int,
    models: Sequence[Any],
    seed: int,
    cv: Any,
    search_type: str,
) -> None:
    """
    Print the AutoML run configuration summary.

    Parameters
    ----------
    task : str
        Type of machine learning task (e.g., classification, regression).
    metric : str
        Evaluation metric used during model selection.
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of input features.
    models : Sequence[Any]
        Collection of candidate models evaluated during the run.
    seed : int
        Random seed used for reproducibility.
    cv : Any
        Cross-validation strategy object. Expected to optionally
        expose an `n_splits` attribute.
    search_type : str
        Hyperparameter search strategy (e.g., grid, random, optuna).

    Returns
    -------
    None
        Prints formatted run metadata to stdout.
    """
    print("\n" + LINE)
    print("AUTO ML RUN".center(CONSOLE_WIDTH))
    print(LINE)

    print(f"Task          : {task}")
    print(f"Metric        : {metric}")
    print(f"Samples       : {n_samples}")
    print(f"Features      : {n_features}")
    print(f"Models        : {len(models)}")
    print(f"Seed          : {seed}")

    # CV description (duck-typed)
    if hasattr(cv, "n_splits"):
        print(
            f"CV Strategy   : {cv.__class__.__name__} "
            f"(folds={cv.n_splits})"
        )

    print(f"Search        : {search_type}")

    print(LINE + "\n")
