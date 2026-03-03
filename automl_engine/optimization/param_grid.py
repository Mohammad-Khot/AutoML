# optimization/param_grid.py

from typing import Dict


# ─────────────────────────────────────────────
# Classification Grid
# ─────────────────────────────────────────────

CLASSIFICATION_PARAM_GRID: Dict[str, dict] = {

    "logistic": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    },

    "ridge": {
        "model__alpha": [0.1, 1.0, 10.0],
    },

    "svm": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale"],
    },

    "knn": {
        "model__n_neighbors": [3, 5, 11, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },

    "naive_bayes": {
        "model__var_smoothing": [1e-9, 1e-8, 1e-7],
    },

    "decision_tree": {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },

    "rf": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__max_features": ["sqrt"],
    },

    "extra_trees": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__max_features": ["sqrt"],
    },

    "gb": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0],
    },

    "hist_gb": {
        "model__max_iter": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [None, 10],
        "model__l2_regularization": [0.0, 0.1],
    },

    "xgboost": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 6],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },

    "dummy": {},
}


# ─────────────────────────────────────────────
# Regression Grid
# ─────────────────────────────────────────────

REGRESSION_PARAM_GRID: Dict[str, dict] = {

    "linear": {},

    "ridge": {
        "model__alpha": [0.1, 1.0, 10.0],
    },

    "lasso": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
    },

    "elastic": {
        "model__alpha": [0.001, 0.01, 0.1],
        "model__l1_ratio": [0.2, 0.5, 0.8],
    },

    "sgd": {
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__penalty": ["l2", "l1"],
        "model__learning_rate": ["optimal", "adaptive"],
    },

    "svm": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale"],
    },

    "knn": {
        "model__n_neighbors": [3, 5, 11, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },

    "decision_tree": {
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },

    "rf": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__max_features": ["sqrt"],
    },

    "extra_trees": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__max_features": ["sqrt"],
    },

    "gb": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0],
    },

    "hist_gb": {
        "model__max_iter": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [None, 10],
        "model__l2_regularization": [0.0, 0.1],
    },

    "xgboost": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 6],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },

    "dummy": {},
}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_param_grid(task: str) -> Dict[str, dict]:

    if task == "classification":
        return CLASSIFICATION_PARAM_GRID

    elif task == "regression":
        return REGRESSION_PARAM_GRID

    else:
        raise ValueError(f"Unknown task: {task}")
