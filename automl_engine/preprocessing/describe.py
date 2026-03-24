from typing import Any, Dict, Union
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Detection imports
from sklearn.feature_selection import SelectorMixin
from sklearn.decomposition import PCA, TruncatedSVD, FastICA


EstimatorLike = Union[BaseEstimator, str, None]


# ─────────────────────────────────────────────
# TYPE DETECTION (SEMANTIC)
# ─────────────────────────────────────────────
FEATURE_ENGINEERING_TYPES = (PCA, TruncatedSVD, FastICA)


def _is_selector(estimator):
    return isinstance(estimator, SelectorMixin)


def _is_feature_engineering(estimator):
    return isinstance(estimator, FEATURE_ENGINEERING_TYPES)


# ─────────────────────────────────────────────
# SAFE ESTIMATOR DESCRIPTION
# ─────────────────────────────────────────────
def _describe_estimator(estimator: EstimatorLike) -> Dict[str, Any]:
    if estimator is None:
        return {"name": "None", "params": {}, "steps": None}

    if isinstance(estimator, str):
        return {"name": estimator, "params": {}, "steps": None}

    name = estimator.__class__.__name__

    info: Dict[str, Any] = {
        "name": name,
        "params": {},
        "steps": None,
    }

    # Params
    try:
        params = estimator.get_params(deep=False)
        info["params"] = {
            k: v
            for k, v in params.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
    except (AttributeError, TypeError):
        info["params"] = {}

    # Pipeline recursion
    if isinstance(estimator, Pipeline):
        info["steps"] = [
            {step_name: _describe_estimator(step_obj)}
            for step_name, step_obj in estimator.steps
        ]
        return info

    # ColumnTransformer recursion
    if isinstance(estimator, ColumnTransformer):
        info["steps"] = [
            {name: _describe_estimator(trans)}
            for name, trans, _ in estimator.transformers
            if trans != "drop"
        ]
        return info

    return info


# ─────────────────────────────────────────────
# MAIN PIPELINE DESCRIBER (SEMANTIC-AWARE)
# ─────────────────────────────────────────────
def describe_pipeline(pipeline: Pipeline) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "transformers": {},
        "feature_engineering": [],
        "selector": None,
        "final_estimator": None,
    }

    for name, step in pipeline.steps:

        try:
            # ───────── COLUMN TRANSFORMER ─────────
            if isinstance(step, ColumnTransformer):

                for transformer_name, transformer, columns in step.transformers:

                    if transformer == "drop":
                        continue

                    summary["transformers"][transformer_name] = {
                        "columns": columns,
                        "details": _describe_estimator(transformer),
                    }

            # ───────── FEATURE SELECTION ─────────
            elif _is_selector(step):
                summary["selector"] = _describe_estimator(step)

            # ───────── FEATURE ENGINEERING ─────────
            elif _is_feature_engineering(step):
                summary["feature_engineering"].append(
                    _describe_estimator(step)
                )

            # ───────── DEFAULT → FINAL MODEL ─────────
            else:
                summary["final_estimator"] = _describe_estimator(step)

        except (AttributeError, TypeError, ValueError) as e:
            summary["transformers"][name] = {
                "columns": [],
                "details": {
                    "name": "ERROR",
                    "params": {"_error": str(e)},
                    "steps": None,
                },
            }

    return summary
