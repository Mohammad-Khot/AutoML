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


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    resolved: ResolvedConfig,
    stage: str,
    cv_override: Any | None = None,
) -> AutoMLState:

    # --- Aliases ---
    runtime = resolved.runtime
    metric = resolved.problem.metric
    models = resolved.artifacts.models

    # 👇 THE FIX
    cv = cv_override if cv_override is not None else resolved.artifacts.cv_object

    state: AutoMLState = AutoMLState()

    # --- Guard against degenerate CV ---
    if hasattr(cv, "n_splits") and getattr(cv, "n_splits") < 2:
        log_model_score(
            "ALL",
            "SKIPPED: insufficient CV folds",
            stage=stage,
            log=runtime.log,
        )
        return state

    # --- Evaluate models ---
    for model_name, model_spec in models.items():

        try:
            pipeline = build_pipeline(
                spec=model_spec,
                resolved=resolved,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)

                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=metric,
                    n_jobs=resolved.runtime.n_jobs,
                )

            scores = np.asarray(scores)
            mean_score: float = float(np.mean(scores))

        except ConvergenceWarning:
            log_model_score(
                model_name,
                "SKIPPED: convergence failure",
                stage=stage,
                log=runtime.log,
            )
            continue

        except Exception as e:
            print("\n" + "=" * 60)
            print(f"[CRASH] Model: {model_name}")
            print("=" * 60)
            raise

        # except Exception as e:
        #     log_model_score(
        #         model_name,
        #         f"ERROR ({type(e).__name__}: {e})",
        #         stage=stage,
        #         log=runtime.log,
        #     )
        #     continue

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
