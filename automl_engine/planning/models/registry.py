# planning/models/registry.py

from types import MappingProxyType

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    RidgeClassifier,
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier

from automl_engine.planning.models.spec import ModelSpec

COST_LOW = "low"
COST_MEDIUM = "medium"
COST_HIGH = "high"


# ---------------------------
# SEARCH SPACES
# ---------------------------

def logistic_space(trial):
    return {
        "model__C": trial.suggest_float("model__C", 1e-4, 100, log=True),
    }


def ridge_classifier_space(trial):
    return {
        "model__alpha": trial.suggest_float("model__alpha", 1e-4, 100, log=True),
    }


def svm_space(trial):
    return {
        "model__C": trial.suggest_float("model__C", 1e-3, 100, log=True),
        "model__gamma": trial.suggest_float("model__gamma", 1e-4, 1, log=True),
        "model__kernel": trial.suggest_categorical(
            "model__kernel", ["rbf", "poly"]
        ),
    }


def knn_space(trial):
    return {
        "model__n_neighbors": trial.suggest_int("model__n_neighbors", 3, 50),
        "model__weights": trial.suggest_categorical(
            "model__weights", ["uniform", "distance"]
        ),
        "model__p": trial.suggest_categorical("model__p", [1, 2]),
    }


def decision_tree_space(trial):
    return {
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 50),
        "model__min_samples_split": trial.suggest_int(
            "model__min_samples_split", 2, 20
        ),
        "model__min_samples_leaf": trial.suggest_int(
            "model__min_samples_leaf", 1, 20
        ),
    }


def rf_space(trial):
    return {
        "model__n_estimators": trial.suggest_int(
            "model__n_estimators", 200, 1200
        ),
        "model__max_depth": trial.suggest_int("model__max_depth", 5, 50),
        "model__min_samples_split": trial.suggest_int(
            "model__min_samples_split", 2, 20
        ),
        "model__min_samples_leaf": trial.suggest_int(
            "model__min_samples_leaf", 1, 20
        ),
        "model__max_features": trial.suggest_categorical(
            "model__max_features", ["sqrt", "log2", None]
        ),
    }


def extra_trees_space(trial):
    return {
        "model__n_estimators": trial.suggest_int(
            "model__n_estimators", 200, 1200
        ),
        "model__max_depth": trial.suggest_int("model__max_depth", 5, 50),
        "model__min_samples_split": trial.suggest_int(
            "model__min_samples_split", 2, 20
        ),
        "model__min_samples_leaf": trial.suggest_int(
            "model__min_samples_leaf", 1, 20
        ),
    }


def gb_space(trial):
    return {
        "model__learning_rate": trial.suggest_float(
            "model__learning_rate", 1e-3, 0.2, log=True
        ),
        "model__n_estimators": trial.suggest_int(
            "model__n_estimators", 100, 1000
        ),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 10),
        "model__subsample": trial.suggest_float(
            "model__subsample", 0.5, 1.0
        ),
    }


def hist_gb_space(trial):
    return {
        "model__learning_rate": trial.suggest_float(
            "model__learning_rate", 1e-3, 0.2, log=True
        ),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 15),
        "model__max_iter": trial.suggest_int("model__max_iter", 100, 1000),
        "model__l2_regularization": trial.suggest_float(
            "model__l2_regularization", 1e-4, 10, log=True
        ),
    }


def xgboost_space(trial):
    return {
        "model__learning_rate": trial.suggest_float(
            "model__learning_rate", 1e-4, 0.2, log=True
        ),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 12),
        "model__subsample": trial.suggest_float(
            "model__subsample", 0.6, 1.0
        ),
        "model__colsample_bytree": trial.suggest_float(
            "model__colsample_bytree", 0.6, 1.0
        ),
        "model__reg_alpha": trial.suggest_float(
            "model__reg_alpha", 1e-8, 10.0, log=True
        ),
        "model__reg_lambda": trial.suggest_float(
            "model__reg_lambda", 1e-8, 10.0, log=True
        ),
        "model__min_child_weight": trial.suggest_float(
            "model__min_child_weight", 1e-2, 10, log=True
        ),
        "model__n_estimators": trial.suggest_int(
            "model__n_estimators", 200, 1500
        ),
    }


def ridge_space(trial):
    return {
        "model__alpha": trial.suggest_float("model__alpha", 1e-4, 100, log=True),
    }


def lasso_space(trial):
    return {
        "model__alpha": trial.suggest_float("model__alpha", 1e-4, 10, log=True),
        "model__max_iter": trial.suggest_int("model__max_iter", 1000, 10000),
    }


def elastic_space(trial):
    return {
        "model__alpha": trial.suggest_float("model__alpha", 1e-4, 10, log=True),
        "model__l1_ratio": trial.suggest_float("model__l1_ratio", 0.05, 0.95),
    }


def sgd_space(trial):
    return {
        "model__alpha": trial.suggest_float("model__alpha", 1e-6, 1e-1, log=True),
        "model__penalty": trial.suggest_categorical(
            "model__penalty", ["l2", "l1", "elasticnet"]
        ),
        "model__learning_rate": trial.suggest_categorical(
            "model__learning_rate", ["constant", "optimal"]
        ),
    }


# ---------------------------
# MODEL REGISTRY
# ---------------------------

MODEL_REGISTRY = MappingProxyType({

    "classification": {

        "logistic": ModelSpec(
            name="logistic",
            factory=lambda: LogisticRegression(max_iter=10000),
            needs_scaling=True,
            supports_sparse=True,
            interpretable=True,
            recommended_for_small_data=True,
            search_space=logistic_space,
            priority=1
        ),

        "ridge": ModelSpec(
            name="ridge",
            factory=lambda: RidgeClassifier(),
            needs_scaling=True,
            supports_sparse=True,
            interpretable=True,
            recommended_for_small_data=True,
            search_space=ridge_classifier_space,
            priority=1
        ),

        "svm": ModelSpec(
            name="svm",
            factory=lambda: SVC(),
            needs_scaling=True,
            supports_sparse=True,
            size_sensitive=True,
            search_space=svm_space,
            priority=2
        ),

        "knn": ModelSpec(
            name="knn",
            factory=lambda: KNeighborsClassifier(),
            needs_scaling=True,
            size_sensitive=True,
            recommended_for_small_data=True,
            search_space=knn_space,
            priority=2
        ),

        "naive_bayes": ModelSpec(
            name="naive_bayes",
            factory=lambda: GaussianNB(),
            recommended_for_small_data=True,
            priority=2
        ),

        "decision_tree": ModelSpec(
            name="decision_tree",
            factory=lambda: DecisionTreeClassifier(),
            supports_sparse=True,
            interpretable=True,
            recommended_for_small_data=True,
            search_space=decision_tree_space,
            priority=2
        ),

        "rf": ModelSpec(
            name="rf",
            factory=lambda: RandomForestClassifier(),
            supports_sparse=True,
            handles_large_datasets=True,
            search_space=rf_space,
            priority=3
        ),

        "extra_trees": ModelSpec(
            name="extra_trees",
            factory=lambda: ExtraTreesClassifier(),
            supports_sparse=True,
            handles_large_datasets=True,
            search_space=extra_trees_space,
            priority=3
        ),

        "gb": ModelSpec(
            name="gb",
            factory=lambda: GradientBoostingClassifier(),
            search_space=gb_space,
            priority=3
        ),

        "hist_gb": ModelSpec(
            name="hist_gb",
            factory=lambda: HistGradientBoostingClassifier(),
            supports_missing=True,
            handles_large_datasets=True,
            search_space=hist_gb_space,
            priority=3
        ),

        "xgboost": ModelSpec(
            name="xgboost",
            factory=lambda: XGBClassifier(),
            supports_missing=True,
            supports_sparse=True,
            supports_gpu=True,
            handles_large_datasets=True,
            search_space=xgboost_space,
            priority=3
        ),

        "dummy": ModelSpec(
            name="dummy",
            factory=lambda: DummyClassifier(strategy="prior"),
            priority=0
        ),
    },


    "regression": {

        "linear": ModelSpec(
            name="linear",
            factory=lambda: LinearRegression(),
            supports_sparse=True,
            interpretable=True,
            recommended_for_small_data=True,
            priority=1
        ),

        "ridge": ModelSpec(
            name="ridge",
            factory=lambda: Ridge(),
            supports_sparse=True,
            interpretable=True,
            recommended_for_small_data=True,
            search_space=ridge_space,
            priority=1
        ),

        "lasso": ModelSpec(
            name="lasso",
            factory=lambda: Lasso(),
            interpretable=True,
            recommended_for_small_data=True,
            search_space=lasso_space,
            priority=1
        ),

        "elastic": ModelSpec(
            name="elastic",
            factory=lambda: ElasticNet(),
            interpretable=True,
            recommended_for_small_data=True,
            search_space=elastic_space,
            priority=1
        ),

        "sgd": ModelSpec(
            name="sgd",
            factory=lambda: SGDRegressor(),
            supports_sparse=True,
            handles_large_datasets=True,
            search_space=sgd_space,
            priority=2
        ),

        "svm": ModelSpec(
            name="svm",
            factory=lambda: SVR(),
            supports_sparse=True,
            size_sensitive=True,
            search_space=svm_space,
            priority=2
        ),

        "knn": ModelSpec(
            name="knn",
            factory=lambda: KNeighborsRegressor(),
            size_sensitive=True,
            search_space=knn_space,
            priority=2
        ),

        "decision_tree": ModelSpec(
            name="decision_tree",
            factory=lambda: DecisionTreeRegressor(),
            supports_sparse=True,
            interpretable=True,
            recommended_for_small_data=True,
            search_space=decision_tree_space,
            priority=2
        ),

        "rf": ModelSpec(
            name="rf",
            factory=lambda: RandomForestRegressor(),
            supports_sparse=True,
            handles_large_datasets=True,
            search_space=rf_space,
            priority=3
        ),

        "extra_trees": ModelSpec(
            name="extra_trees",
            factory=lambda: ExtraTreesRegressor(),
            supports_sparse=True,
            handles_large_datasets=True,
            search_space=extra_trees_space,
            priority=3
        ),

        "gb": ModelSpec(
            name="gb",
            factory=lambda: GradientBoostingRegressor(),
            search_space=gb_space,
            priority=3
        ),

        "hist_gb": ModelSpec(
            name="hist_gb",
            factory=lambda: HistGradientBoostingRegressor(),
            supports_missing=True,
            handles_large_datasets=True,
            search_space=hist_gb_space,
            priority=3
        ),

        "xgboost": ModelSpec(
            name="xgboost",
            factory=lambda: XGBRegressor(),
            supports_missing=True,
            supports_sparse=True,
            supports_gpu=True,
            handles_large_datasets=True,
            search_space=xgboost_space,
            priority=3
        ),

        "dummy": ModelSpec(
            name="dummy",
            factory=lambda: DummyRegressor(strategy="mean"),
            priority=0
        ),
    }

})


def get_model(task: str, name: str) -> BaseEstimator:
    """
    Retrieve and instantiate a model from the MODEL_REGISTRY.

    Args:
        task (str): The task type ("classification" or "regression").
        name (str): The model name registered under the given task.

    Returns:
        BaseEstimator: A new instance of the requested sklearn-compatible estimator.

    Raises:
        ValueError: If the task or model name is not found in the registry.
    """
    try:
        spec: ModelSpec = MODEL_REGISTRY[task][name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}' for task '{task}'.") from exc

    return spec.factory()
