# planning/models/registry.py

from types import MappingProxyType

from sklearn.base import BaseEstimator
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
from xgboost import XGBRegressor, XGBClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier

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
            "model": lambda: LogisticRegression(max_iter=10_000),
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "ridge": {
            **BASE_META,
            "model": lambda: RidgeClassifier(),
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "svm": {
            **BASE_META,
            "model": lambda: SVC(),
            "needs_scaling": True,
            "size_sensitive": True,

        },
        "knn": {
            **BASE_META,
            "model": lambda: KNeighborsClassifier(),
            "needs_scaling": True,
            "size_sensitive": True,

        },
        "naive_bayes": {
            **BASE_META,
            "model": lambda: GaussianNB(),
        },
        "decision_tree": {
            **BASE_META,
            "model": lambda: DecisionTreeClassifier(),
            "handles_high_dim": True,
        },
        "rf": {
            **BASE_META,
            "model": lambda: RandomForestClassifier(),
            "compute_cost": COST_HIGH,
            "handles_high_dim": True,

        },
        "extra_trees": {
            **BASE_META,
            "model": lambda: ExtraTreesClassifier(),
            "compute_cost": COST_HIGH

        },
        "gb": {
            **BASE_META,
            "model": lambda: GradientBoostingClassifier(),
            "compute_cost": COST_HIGH

        },
        "hist_gb": {
            **BASE_META,
            "model": lambda: HistGradientBoostingClassifier(),
            "compute_cost": COST_HIGH,
            "native_categorical": True
        },
        "xgboost": {
            **BASE_META,
            "model": lambda: XGBClassifier(),
            "compute_cost": COST_HIGH,
            "handles_high_dim": True
        },
        # "catboost": {
        #     **BASE_META,
        #     "model": lambda: CatBoostClassifier(verbose=False),
        #     "compute_cost": COST_MEDIUM,
        #     "handles_high_dim": True,
        #     "native_categorical": True,
        # },
        # "lightgbm": {
        #     **BASE_META,
        #     "model": lambda: LGBMClassifier(),
        #     "compute_cost": COST_HIGH,
        #     "handles_high_dim": True,
        #     "native_categorical": True,
        # },
        "dummy": {
            **BASE_META,
            "model": lambda: DummyClassifier(strategy="prior"),
        }
    },
    "regression": {
        "linear": {
            **BASE_META,
            "model": lambda: LinearRegression(),
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "ridge": {
            **BASE_META,
            "model": lambda: Ridge(),
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "lasso": {
            **BASE_META,
            "model": lambda: Lasso(),
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "elastic": {
            **BASE_META,
            "model": lambda: ElasticNet(),
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "sgd": {
            **BASE_META,
            "model": lambda: SGDRegressor(),
            "needs_scaling": True,
            "interpretable": True,
            "compute_cost": COST_LOW
        },
        "svm": {
            **BASE_META,
            "model": lambda: SVR(),
            "needs_scaling": True,
            "size_sensitive": True,
        },
        "knn": {
            **BASE_META,
            "model": lambda: KNeighborsRegressor(),
            "needs_scaling": True,
            "size_sensitive": True,
        },
        "decision_tree": {
            **BASE_META,
            "model": lambda: DecisionTreeRegressor(),
            "handles_high_dim": True,
        },
        "rf": {
            **BASE_META,
            "model": lambda: RandomForestRegressor(),
            "compute_cost": COST_HIGH,
            "handles_high_dim": True,
        },
        "extra_trees": {
            **BASE_META,
            "model": lambda: ExtraTreesRegressor(),
            "compute_cost": COST_HIGH
        },
        "gb": {
            **BASE_META,
            "model": lambda: GradientBoostingRegressor(),
            "compute_cost": COST_HIGH
        },
        "hist_gb": {
            **BASE_META,
            "model": lambda: HistGradientBoostingRegressor(),
            "compute_cost": COST_HIGH,
            "native_categorical": True,
        },
        "xgboost": {
            **BASE_META,
            "model": lambda: XGBRegressor(),
            "compute_cost": COST_HIGH,
            "handles_high_dim": True
        },
        # "catboost": {
        #     **BASE_META,
        #     "model": lambda: CatBoostRegressor(verbose=False),
        #     "compute_cost": COST_MEDIUM,
        #     "handles_high_dim": True,
        #     "native_categorical": True,
        # },
        # "lightgbm": {
        #     **BASE_META,
        #     "model": lambda: LGBMRegressor(),
        #     "compute_cost": COST_HIGH,
        #     "handles_high_dim": True,
        #     "native_categorical": True,
        # },
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
    "xgboost": 3,
    # "catboost": 3,
    # "lightgbm": 3,
}


def get_model(task: str, name: str) -> BaseEstimator:
    """
    Retrieve and instantiate a model from the registry.

    Parameters
    ----------
    task : str
        The task type. Must be either "classification" or "regression".
    name : str
        The model identifier defined in MODEL_REGISTRY for the given task.

    Returns
    -------
    BaseEstimator
        A newly instantiated scikit-learn compatible estimator.

    Raises
    ------
    ValueError
        If the task or model name does not exist in the registry.
    """
    try:
        model_factory = MODEL_REGISTRY[task][name]["model"]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}' for task '{task}'.") from exc

    return model_factory()
