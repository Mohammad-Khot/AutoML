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

COST_LOW = "low"
COST_MEDIUM = "medium"
COST_HIGH = "high"

BASE_META = {
    "needs_scaling": False,
    "size_sensitive": False,
    "handles_high_dim": False,
    "native_categorical": False,
    "interpretable": False,
    "compute_cost": COST_MEDIUM
}


# ---------------------------
# SEARCH SPACE DEFINITIONS
# ---------------------------

# ===== CLASSIFICATION =====

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


# ===== REGRESSION =====

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
# REGISTRY
# ---------------------------

MODEL_REGISTRY = MappingProxyType({
    "classification": {
        "logistic": {**BASE_META, "model": lambda: LogisticRegression(max_iter=10_000), "needs_scaling": True,
                     "search_space": logistic_space},
        "ridge": {**BASE_META, "model": lambda: RidgeClassifier(), "needs_scaling": True,
                  "search_space": ridge_classifier_space},
        "svm": {**BASE_META, "model": lambda: SVC(), "needs_scaling": True, "search_space": svm_space},
        "knn": {**BASE_META, "model": lambda: KNeighborsClassifier(), "needs_scaling": True, "search_space": knn_space},
        "naive_bayes": {**BASE_META, "model": lambda: GaussianNB()},
        "decision_tree": {**BASE_META, "model": lambda: DecisionTreeClassifier(), "search_space": decision_tree_space},
        "rf": {**BASE_META, "model": lambda: RandomForestClassifier(), "search_space": rf_space},
        "extra_trees": {**BASE_META, "model": lambda: ExtraTreesClassifier(), "search_space": extra_trees_space},
        "gb": {**BASE_META, "model": lambda: GradientBoostingClassifier(), "search_space": gb_space},
        "hist_gb": {**BASE_META, "model": lambda: HistGradientBoostingClassifier(), "search_space": hist_gb_space},
        "xgboost": {**BASE_META, "model": lambda: XGBClassifier(), "search_space": xgboost_space},
        "dummy": {**BASE_META, "model": lambda: DummyClassifier(strategy="prior")},
    },
    "regression": {
        "linear": {**BASE_META, "model": lambda: LinearRegression()},
        "ridge": {**BASE_META, "model": lambda: Ridge(), "search_space": ridge_space},
        "lasso": {**BASE_META, "model": lambda: Lasso(), "search_space": lasso_space},
        "elastic": {**BASE_META, "model": lambda: ElasticNet(), "search_space": elastic_space},
        "sgd": {**BASE_META, "model": lambda: SGDRegressor(), "search_space": sgd_space},
        "svm": {**BASE_META, "model": lambda: SVR(), "search_space": svm_space},
        "knn": {**BASE_META, "model": lambda: KNeighborsRegressor(), "search_space": knn_space},
        "decision_tree": {**BASE_META, "model": lambda: DecisionTreeRegressor(), "search_space": decision_tree_space},
        "rf": {**BASE_META, "model": lambda: RandomForestRegressor(), "search_space": rf_space},
        "extra_trees": {**BASE_META, "model": lambda: ExtraTreesRegressor(), "search_space": extra_trees_space},
        "gb": {**BASE_META, "model": lambda: GradientBoostingRegressor(), "search_space": gb_space},
        "hist_gb": {**BASE_META, "model": lambda: HistGradientBoostingRegressor(), "search_space": hist_gb_space},
        "xgboost": {**BASE_META, "model": lambda: XGBRegressor(), "search_space": xgboost_space},
        "dummy": {**BASE_META, "model": lambda: DummyRegressor(strategy="mean")},
    },
})

MODEL_PRIORITY = {

    # ----------Classifiers----------
    "logistic": 1,

    "naive_bayes": 2,

    # ----------Regressors----------
    "linear": 1,
    "lasso": 1,
    "elastic": 1,

    "sgd": 2,

    # ----------Classifiers & Regressors----------
    "dummy": 0,

    "ridge": 1,

    "knn": 2,
    "decision_tree": 2,
    "svm": 2,

    "rf": 3,
    "extra_trees": 3,
    "gb": 3,
    "hist_gb": 3,
    "xgboost": 3,
    # "catboost": 3,
    # "lightgbm": 3,
}


def get_model(task: str, name: str) -> BaseEstimator:
    try:
        model_factory = MODEL_REGISTRY[task][name]["model"]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}' for task '{task}'.") from exc

    return model_factory()
