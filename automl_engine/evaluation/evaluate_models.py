# evaluation/evaluate_models.py

from typing import Any
import numpy as np
import warnings

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning

from automl_engine.planning.experiment.resolved import ResolvedConfig
from automl_engine.reporting import log_model_score
from automl_engine.preprocessing import build_pipeline
from automl_engine.runtime import AutoMLState
from automl_engine.evaluation.metrics import get_scorer_safe


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig,
    stage: str,
    cv_override: Any | None = None,
) -> AutoMLState:
    runtime = resolved.runtime
    models = resolved.artifacts.models
    scorer = get_scorer_safe(resolved.problem.metric)

    cv = cv_override if cv_override is not None else resolved.artifacts.cv_object
    state = AutoMLState()

    if hasattr(cv, "n_splits") and getattr(cv, "n_splits") < 2:
        log_model_score(
            "ALL",
            "SKIPPED: insufficient CV folds",
            stage=stage,
            log=runtime.log,
        )
        return state

    for model_name, model_spec in models.items():
        try:
            pipeline = build_pipeline(spec=model_spec, resolved=resolved)

            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=resolved.runtime.n_jobs,
                )

            scores = np.asarray(scores)
            mean_score = float(np.mean(scores))

            if not np.isfinite(mean_score):
                log_model_score(
                    model_name,
                    "SKIPPED: non-finite score",
                    stage=stage,
                    log=runtime.log,
                )
                continue

        except ConvergenceWarning:
            log_model_score(
                model_name,
                "SKIPPED: convergence failure",
                stage=stage,
                log=runtime.log,
            )
            continue

        except Exception as exc:
            # A single incompatible estimator must not terminate an AutoML run.
            log_model_score(
                model_name,
                f"ERROR ({type(exc).__name__}: {exc})",
                stage=stage,
                log=runtime.log,
            )
            continue

        log_model_score(
            model_name,
            round(mean_score, 4),
            stage=stage,
            log=runtime.log,
        )

        state.update(
            model_name,
            mean_score,
            pipeline=pipeline,
            params=None,
        )

    return state
