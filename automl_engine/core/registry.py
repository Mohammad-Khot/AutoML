# core/registry.py

from types import MappingProxyType

from sklearn.linear_model import (
    RidgeClassifier,
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor
)
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
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier

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

MODEL_REGISTRY = MappingProxyType({
    "classification": {
        "logistic": {
            **BASE_META,
            "model": LogisticRegression,
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "ridge": {
            **BASE_META,
            "model": RidgeClassifier,
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "svm": {
            **BASE_META,
            "model": SVC,
            "needs_scaling": True,
            "size_sensitive": True,

        },
        "knn": {
            **BASE_META,
            "model": KNeighborsClassifier,
            "needs_scaling": True,
            "size_sensitive": True,

        },
        "naive_bayes": {
            **BASE_META,
            "model": GaussianNB,
        },
        "decision_tree": {
            **BASE_META,
            "model": DecisionTreeClassifier,
            "handles_high_dim": True,
        },
        "rf": {
            **BASE_META,
            "model": RandomForestClassifier,
            "compute_cost": COST_HIGH,
            "handles_high_dim": True,

        },
        "extra_trees": {
            **BASE_META,
            "model": ExtraTreesClassifier,
            "compute_cost": COST_HIGH

        },
        "gb": {
            **BASE_META,
            "model": GradientBoostingClassifier,
            "compute_cost": COST_HIGH

        },
        "hist_gb": {
            **BASE_META,
            "model": HistGradientBoostingClassifier,
            "compute_cost": COST_HIGH,
            "native_categorical": True
        },
        "dummy": {
            **BASE_META,
            "model": lambda: DummyClassifier(strategy="prior"),
        }
    },
    "regression": {
        "linear": {
            **BASE_META,
            "model": LinearRegression,
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "ridge": {
            **BASE_META,
            "model": Ridge,
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "lasso": {
            **BASE_META,
            "model": Lasso,
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "elastic": {
            **BASE_META,
            "model": ElasticNet,
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "sgd": {
            **BASE_META,
            "model": SGDRegressor,
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "svm": {
            **BASE_META,
            "model": SVR,
            "needs_scaling": True,
            "size_sensitive": True,

        },
        "knn": {
            **BASE_META,
            "model": KNeighborsRegressor,
            "needs_scaling": True,
            "size_sensitive": True,
        },
        "decision_tree": {
            **BASE_META,
            "model": DecisionTreeRegressor,
            "handles_high_dim": True,

        },
        "rf": {
            **BASE_META,
            "model": RandomForestRegressor,
            "compute_cost": COST_HIGH,
            "handles_high_dim": True,

        },
        "extra_trees": {
            **BASE_META,
            "model": ExtraTreesRegressor,
            "compute_cost": COST_HIGH

        },
        "gb": {
            **BASE_META,
            "model": GradientBoostingRegressor,
            "compute_cost": COST_HIGH

        },
        "hist_gb": {
            **BASE_META,
            "model": HistGradientBoostingRegressor,
            "compute_cost": COST_HIGH,
            "native_categorical": True,

        },
        "dummy": {
            **BASE_META,
            "model": lambda: DummyRegressor(strategy="mean"),
        }
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

}


def get_model(task: str, name: str):
    try:
        factory = MODEL_REGISTRY[task][name]["model"]
        return factory() if callable(factory) else factory()
    except KeyError:
        raise ValueError(f"Unknown model {name} for task {task}")


for task in MODEL_REGISTRY:
    for name in MODEL_REGISTRY[task]:
        if name not in MODEL_PRIORITY:
            raise ValueError(f"Priority missing for model: {name}")
