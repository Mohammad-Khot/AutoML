# evaluation/evaluate_models.py

from typing import Any, Dict
import numpy as np
from sklearn.model_selection import cross_val_score

from automl_engine.reporting import log_model_score
from automl_engine.preprocessing import build_pipeline
from automl_engine.runtime.state import AutoMLState


def evaluate_models(
    X: Any,
    y: Any,
    models: Dict[str, Any],
    cv: Any,
    config: Any,
    resolved: Any,
    stage: str,
) -> AutoMLState:
    """
    Evaluate candidate models using cross-validation.
    Pure evaluation only — no hyperparameter search.
    """

    state: AutoMLState = AutoMLState()

    # ---------- CV sanity ----------
    if hasattr(cv, "n_splits") and getattr(cv, "n_splits") < 2:
        log_model_score(
            "ALL",
            "SKIPPED: insufficient CV folds",
            stage=stage,
            log=config.log,
        )
        return state

    metric: str = resolved.metric

    # ---------- Model loop ----------
    for name, info in models.items():
        try:
            pipeline: Any = build_pipeline(
                info,
                X,
                config,
                seed=config.seed,
            )

            scores: np.ndarray = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=metric,
                n_jobs=config.n_jobs
            )

            mean_score: float = float(np.mean(scores))
            best_params = None

        except Exception as e:
            log_model_score(
                name,
                f"ERROR ({type(e).__name__}: {e})",
                stage=stage,
                log=config.log,
            )
            continue

        # ---------- Logging ----------
        log_model_score(
            name,
            round(mean_score, 4),
            stage=stage,
            log=config.log,
        )

        state.update(
            name,
            mean_score,
            pipeline=pipeline,
            params=best_params,
        )

    return state
