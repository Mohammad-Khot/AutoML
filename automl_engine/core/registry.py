# core/registry.py

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

MODEL_REGISTRY = {
    "classification": {
        "logistic": {
            "model": lambda: LogisticRegression(),
            "needs_scaling": True
        },
        "ridge": {
            "model": lambda: RidgeClassifier(),
            "needs_scaling": True
        },
        "svm": {
            "model": lambda: SVC(
                kernel="rbf",
                probability=False,
                cache_size=200
            ),
            "needs_scaling": True
        },
        "knn": {
            "model": lambda: KNeighborsClassifier(),
            "needs_scaling": True
        },
        "naive_bayes": {
            "model": lambda: GaussianNB(),
            "needs_scaling": False
        },
        "decision_tree": {
            "model": lambda: DecisionTreeClassifier(
                max_depth=15,
                min_samples_leaf=2
            ),
            "needs_scaling": False
        },
        "rf": {
            "model": lambda: RandomForestClassifier(),
            "needs_scaling": False
        },
        "extra_trees": {
            "model": lambda: ExtraTreesClassifier(),
            "needs_scaling": False
        },
        "gb": {
            "model": lambda: GradientBoostingClassifier(),
            "needs_scaling": False
        },
        "hist_gb": {
            "model": lambda: HistGradientBoostingClassifier(),
            "needs_scaling": False
        },
        "dummy": {
            "model": lambda: DummyClassifier(strategy="prior"),
            "needs_scaling": False
        }
    },
    "regression": {
        "linear": {
            "model": lambda: LinearRegression(),
            "needs_scaling": True
        },
        "ridge": {
            "model": lambda: Ridge(),
            "needs_scaling": True
        },
        "lasso": {
            "model": lambda: Lasso(),
            "needs_scaling": True
        },
        "elastic": {
            "model": lambda: ElasticNet(),
            "needs_scaling": True
        },
        "sgd": {
            "model": lambda: SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                max_iter=2000,
                tol=1e-3,
                learning_rate="adaptive",
                eta0=0.01
            ),
            "needs_scaling": True
        },
        "svm": {
            "model": lambda: SVR(
                C=1.0,
                epsilon=0.1,
                cache_size=200
            ),
            "needs_scaling": True,
            "size_sensitive": True
        },
        "knn": {
            "model": lambda: KNeighborsRegressor(
                n_neighbors=5,
                weights="distance",
                algorithm="auto"
            ),
            "needs_scaling": True
        },
        "decision_tree": {
            "model": lambda: DecisionTreeRegressor(
                max_depth=15,
                min_samples_leaf=2
            ),
            "needs_scaling": False
        },
        "rf": {
            "model": lambda: RandomForestRegressor(
                n_estimators=80,
                max_depth=20,
                min_samples_leaf=2,
                n_jobs=1
            ),
            "needs_scaling": False
        },
        "extra_trees": {
            "model": lambda: ExtraTreesRegressor(
                n_estimators=80,
                max_depth=20,
                n_jobs=1
            ),
            "needs_scaling": False
        },
        "gb": {
            "model": lambda: GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=3,
                subsample=0.85
            ),
            "needs_scaling": False
        },
        "hist_gb": {
            "model": lambda: HistGradientBoostingRegressor(
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8
            ),
            "needs_scaling": False
        },
        "dummy": {
            "model": lambda: DummyRegressor(strategy="mean"),
            "needs_scaling": False
        }
    },
}

MODEL_PRIORITY = {

    # ----------Classifiers----------
    "logistic": 1,
    "svm": 2,

    "naive_bayes": 2,

    # ----------Regressors----------
    "linear": 1,
    "lasso": 1,
    "elastic": 1,

    "svr": 2,
    "sgd": 2,

    # ----------Classifiers & Regressors----------
    "dummy": 0,

    "ridge": 1,

    "knn": 2,
    "decision_tree": 2,

    "rf": 3,
    "extra_trees": 3,
    "gb": 3,
    "hist_gb": 3,

}
