# automl_engine/reporting/run_report.py
import logging
from typing import Final

from .console import CONSOLE_WIDTH

LINE: Final[str] = "=" * CONSOLE_WIDTH

logger = logging.getLogger(__name__)


def print_run_header(X, y, resolved) -> None:
    print("\n" + LINE)
    print("AUTO ML RUN".center(CONSOLE_WIDTH))
    print(LINE)

    problem = getattr(resolved, "problem", None)
    runtime = getattr(resolved, "runtime", None)
    artifacts = getattr(resolved, "artifacts", None)

    task = getattr(problem, "task", "N/A")
    metric = getattr(problem, "metric", "N/A")
    seed = getattr(runtime, "seed", "N/A")

    print(f"Task          : {str(task).capitalize()}")
    print(f"Metric        : {str(metric).capitalize()}")
    print(f"Samples       : {len(X)}")
    print(f"Features      : {X.shape[1]}")
    print(f"Seed          : {seed}")

    # ─────────────────────────────
    # Models
    # ─────────────────────────────
    models = getattr(artifacts, "models", {}) or {}

    n_models = safe_call(lambda: len(models), context="len(models)")
    model_names = list(models.keys())

    rows = format_models_grid(model_names)

    print_block(
        label="Models        : ",
        first_value=str(n_models),
        rows=rows,
    )

    # ─────────────────────────────
    # CV Strategy
    # ─────────────────────────────
    cv = getattr(artifacts, "cv_object", None)

    if cv is not None and hasattr(cv, "n_splits"):
        extra = []

        if hasattr(cv, "shuffle"):
            extra.append(f"shuffle={cv.shuffle}")

        extra_str = f", {', '.join(extra)}" if extra else ""

        print(
            f"CV Strategy   : {cv.__class__.__name__} "
            f"({cv.n_splits} folds{extra_str})"
        )
    else:
        print("CV Strategy   : N/A")

    # ─────────────────────────────
    # Dataset diagnostics
    # ─────────────────────────────
    missing = safe_call(
        lambda: int(X.isna().sum().sum()),
        context="missing_values_count"
    )
    if missing:
        print(f"Missing vals  : {missing}")

    if str(task).lower() == "classification":
        dist = safe_call(
            lambda: y.value_counts(normalize=True).to_dict(),
            context="class_distribution"
        )
        print(f"Class dist    : {format_class_dist(dist)}")

    # ─────────────────────────────
    # Config
    # ─────────────────────────────
    config = getattr(resolved, "config", None)

    if config is not None:
        print(f"Time budget S.: {getattr(config, 'time_budget_soft', 'N/A')}")
        print(f"Tuning        : {getattr(config, 'tuner', 'N/A')}")
        print(f"Early stop    : {getattr(config, 'early_stopping', 'N/A')}")

    print(LINE + "\n")


def safe_call(fn, fallback="N/A", context: str = ""):
    try:
        return fn()
    except Exception as e:
        logger.exception(
            f"[safe_call] Failed in {context or fn.__name__}: {e}"
        )
        return f"{fallback} ({type(e).__name__})"


def get_model_name(m):
    return safe_call(
        lambda: (
            m if isinstance(m, str)
            else m.__name__ if hasattr(m, "__name__")
            else m.__class__.__name__
        ),
        fallback="<model_error>"
    )


def format_class_dist(dist):
    if isinstance(dist, str):
        return dist

    return ", ".join(
        f"{cls} ({p * 100:.1f}%)"
        for cls, p in dist.items()
    )


def format_models_grid(model_names, cols=4, col_width=15):
    rows = []
    for i in range(0, len(model_names), cols):
        chunk = model_names[i:i + cols]
        row = "".join(name.ljust(col_width) for name in chunk)
        rows.append(row.rstrip())
    return rows


def print_block(label: str, first_value: str, rows: list[str]):
    indent = " " * len(label)

    print(f"{label}{first_value}")
    for row in rows:
        print(f"{indent}{row}")
