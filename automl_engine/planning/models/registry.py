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
        "model__C": trial.suggest_float("model__C", 1e-3, 10, log=True),
        "model__l1_ratio": trial.suggest_float("model__l1_ratio", 0.0, 1.0),
    }


def ridge_space(trial):
    return {
        "model__alpha": trial.suggest_float("model__alpha", 1e-4, 1e4, log=True),
    }


def gaussian_nb_space(trial):
    return {
        # Widening the smoothing can sometimes improve generalization
        "model__var_smoothing": trial.suggest_float("model__var_smoothing", 1e-11, 1e-2, log=True)
    }


def svm_space(trial):
    kernel = trial.suggest_categorical("model__kernel", ["linear", "rbf", "poly"])
    params = {
        "model__C": trial.suggest_float("model__C", 1e-3, 100, log=True),
        "model__kernel": kernel,
    }
    if kernel in ["rbf", "poly"]:
        params["model__gamma"] = trial.suggest_float("model__gamma", 1e-4, 1.0, log=True)
    if kernel == "poly":
        params["model__degree"] = trial.suggest_int("model__degree", 2, 5)
    return params


def knn_space(trial):
    return {
        "model__n_neighbors": trial.suggest_int("model__n_neighbors", 3, 50),
        "model__weights": trial.suggest_categorical("model__weights", ["uniform", "distance"]),
        "model__p": trial.suggest_categorical("model__p", [1, 2]),
    }


def decision_tree_space(trial):
    return {
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 50),
        "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
        "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 20),
    }


def random_forest_space(trial):
    return {
        # Added step=100 so Optuna doesn't waste time testing 201 vs 202 estimators
        "model__n_estimators": trial.suggest_int("model__n_estimators", 200, 1200, step=100),
        "model__max_depth": trial.suggest_int("model__max_depth", 5, 50),
        "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
        "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 20),
        "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", "log2", None]),
    }


def extra_trees_space(trial):
    return {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 200, 1200, step=100),
        "model__max_depth": trial.suggest_int("model__max_depth", 5, 50),
        "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
        "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 20),
        "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", "log2", None]),
    }


def gb_space(trial):
    return {
        "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.2, log=True),
        "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 1000, step=100),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 10),
        "model__subsample": trial.suggest_float("model__subsample", 0.5, 1.0),
        "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", "log2", None]),
    }


def hist_gb_space(trial):
    return {
        "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.2, log=True),
        "model__max_iter": trial.suggest_int("model__max_iter", 100, 1000, step=100),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 15),
        "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 10, 50),  # Great for HistGBM
        "model__l2_regularization": trial.suggest_float("model__l2_regularization", 1e-4, 10, log=True),
    }


def xgboost_space(trial):
    return {
        "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-4, 0.3, log=True),
        "model__n_estimators": trial.suggest_int("model__n_estimators", 200, 1500, step=100),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 12),
        "model__min_child_weight": trial.suggest_float("model__min_child_weight", 1e-2, 10, log=True),
        "model__gamma": trial.suggest_float("model__gamma", 1e-4, 10.0, log=True),  # Added gamma
        "model__subsample": trial.suggest_float("model__subsample", 0.5, 1.0),
        "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.5, 1.0),
        "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10.0, log=True),
        "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True),
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
        "model__max_iter": trial.suggest_int("model__max_iter", 1000, 10000),
    }


def sgd_space(trial):
    penalty = trial.suggest_categorical("model__penalty", ["l2", "l1", "elasticnet"])
    learning_rate = trial.suggest_categorical("model__learning_rate", ["constant", "optimal"])

    params = {
        "model__alpha": trial.suggest_float("model__alpha", 1e-6, 1e-1, log=True),
        "model__penalty": penalty,
        "model__learning_rate": learning_rate,
    }

    if penalty == "elasticnet":
        params["model__l1_ratio"] = trial.suggest_float("model__l1_ratio", 0.0, 1.0)

    if learning_rate == "constant":
        params["model__eta0"] = trial.suggest_float("model__eta0", 1e-4, 1e-1, log=True)

    return params


def empty_space():
    # Used for Dummy models and standard Linear Regression
    return {}


# ---------------------------
# MODEL REGISTRY
# ---------------------------

MODEL_REGISTRY = MappingProxyType({
    "classification": {
        "logistic_regression": ModelSpec(
            name="Logistic Regression",
            factory=lambda: LogisticRegression(solver="saga", max_iter=10_000),

            family="linear",

            requires_scaling=True,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=False,

            tuning_complexity="low",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=logistic_space,
            tie_breaker_score=1
        ),
        "ridge": ModelSpec(
            name="Ridge Classifier",
            factory=lambda: RidgeClassifier(),

            family="linear",

            requires_scaling=True,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=False,

            tuning_complexity="low",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=ridge_space,
            tie_breaker_score=2
        ),
        "svc": ModelSpec(
            name="Support Vector Classifier",
            factory=lambda: SVC(),

            family="kernel",

            requires_scaling=True,
            sensitive_to_dataset_size=True,

            captures_feature_interactions=True,
            feature_engineering_impact="medium",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="high",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=False,
            suitable_for_small_datasets=True,

            hyperparameter_space=svm_space,
            tie_breaker_score=4
        ),
        "knn": ModelSpec(
            name="K-Nearest Neighbors Classifier",
            factory=lambda: KNeighborsClassifier(),

            family="distance",

            requires_scaling=True,
            sensitive_to_dataset_size=True,

            captures_feature_interactions=True,
            feature_engineering_impact="medium",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="medium",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=False,
            suitable_for_small_datasets=True,

            hyperparameter_space=knn_space,
            tie_breaker_score=5
        ),
        "Gaussian nb": ModelSpec(
            name="Gaussian Naive Bayes",
            factory=lambda: GaussianNB(),

            family="baseline",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=True,

            tuning_complexity="low",

            training_cost="low",

            supports_sparse_input=False,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=gaussian_nb_space,
            tie_breaker_score=3
        ),
        "decision tree": ModelSpec(
            name="Decision Tree Classifier",
            factory=lambda: DecisionTreeClassifier(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=decision_tree_space,
            tie_breaker_score=4
        ),
        "random forest": ModelSpec(
            name="Random Forest Classifier",
            factory=lambda: RandomForestClassifier(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="medium",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=random_forest_space,
            tie_breaker_score=2
        ),
        "extra trees": ModelSpec(
            name="Extra Trees Classifier",
            factory=lambda: ExtraTreesClassifier(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="medium",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=extra_trees_space,
            tie_breaker_score=3
        ),
        "grad boost": ModelSpec(
            name="Gradient Boosting Classifier",
            factory=lambda: GradientBoostingClassifier(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=True,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="high",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=False,
            suitable_for_small_datasets=True,

            hyperparameter_space=gb_space,
            tie_breaker_score=3
        ),
        "hist grad boost": ModelSpec(
            name="Histogram Gradient Boosting Classifier",
            factory=lambda: HistGradientBoostingClassifier(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="medium",

            supports_sparse_input=False,
            handles_missing_values=True,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=False,

            hyperparameter_space=hist_gb_space,
            tie_breaker_score=1
        ),
        "xgboost": ModelSpec(
            name="XGBoost Classifier",
            factory=lambda: XGBClassifier(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=True,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=False,

            hyperparameter_space=xgboost_space,
            tie_breaker_score=1
        ),
        "dummy": ModelSpec(
            name="Dummy Classifier",
            factory=lambda: DummyClassifier(),

            family="baseline",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=False,

            tuning_complexity="low",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=True,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=empty_space,
            tie_breaker_score=99
        )
    },
    "regression": {
        "linear regression": ModelSpec(
            name="Linear Regression",
            factory=lambda: LinearRegression(),

            family="linear",

            requires_scaling=True,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=False,

            tuning_complexity="low",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=empty_space,
            tie_breaker_score=1
        ),
        "ridge regression": ModelSpec(
            name="Ridge Regression",
            factory=lambda: Ridge(),

            family="linear",

            requires_scaling=True,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=False,

            tuning_complexity="medium",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=ridge_space,
            tie_breaker_score=2
        ),
        "lasso regression": ModelSpec(
            name="Lasso Regression",
            factory=lambda: Lasso(),

            family="linear",

            requires_scaling=True,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=False,

            tuning_complexity="medium",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=lasso_space,
            tie_breaker_score=2
        ),
        "elasticnet": ModelSpec(
            name="ElasticNet",
            factory=lambda: ElasticNet(),

            family="linear",

            requires_scaling=True,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=False,

            tuning_complexity="medium",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=elastic_space,
            tie_breaker_score=3
        ),
        "sgd": ModelSpec(
            name="Stochastic Gradient Descent Regressor",
            factory=lambda: SGDRegressor(),

            family="linear",

            requires_scaling=True,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="high",
            prefers_log_transformed_features=True,
            supports_nonlinearity=False,

            tuning_complexity="high",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=False,

            hyperparameter_space=sgd_space,
            tie_breaker_score=4
        ),
        "svm": ModelSpec(
            name="Support Vector Regressor",
            factory=lambda: SVR(),

            family="kernel",

            requires_scaling=True,
            sensitive_to_dataset_size=True,

            captures_feature_interactions=True,
            feature_engineering_impact="medium",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="high",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=False,
            suitable_for_small_datasets=True,

            hyperparameter_space=svm_space,
            tie_breaker_score=5
        ),
        "knn": ModelSpec(
            name="K-Nearest Neighbors Regressor",
            factory=lambda: KNeighborsRegressor(),

            family="distance",

            requires_scaling=True,
            sensitive_to_dataset_size=True,

            captures_feature_interactions=True,
            feature_engineering_impact="medium",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="medium",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=False,
            suitable_for_small_datasets=True,

            hyperparameter_space=knn_space,
            tie_breaker_score=6
        ),
        "decision tree": ModelSpec(
            name="Decision Tree Regressor",
            factory=lambda: DecisionTreeRegressor(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=decision_tree_space,
            tie_breaker_score=5
        ),
        "random_forest": ModelSpec(
            name="Random Forest Regressor",
            factory=lambda: RandomForestRegressor(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="medium",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=random_forest_space,
            tie_breaker_score=2
        ),
        "extra trees": ModelSpec(
            name="Extra Trees Regressor",
            factory=lambda: ExtraTreesRegressor(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="medium",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=extra_trees_space,
            tie_breaker_score=3
        ),
        "grad boost": ModelSpec(
            name="Gradient Boosting Regressor",
            factory=lambda: GradientBoostingRegressor(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=True,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="high",

            supports_sparse_input=True,
            handles_missing_values=False,

            scales_to_large_datasets=False,
            suitable_for_small_datasets=True,

            hyperparameter_space=gb_space,
            tie_breaker_score=3
        ),
        "hist grad boost": ModelSpec(
            name="Histogram Gradient Boosting Regressor",
            factory=lambda: HistGradientBoostingRegressor(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="medium",

            supports_sparse_input=False,
            handles_missing_values=True,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=False,

            hyperparameter_space=hist_gb_space,
            tie_breaker_score=1
        ),
        "xgboost": ModelSpec(
            name="XGBoost Regressor",
            factory=lambda: XGBRegressor(),

            family="tree",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=True,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=True,

            tuning_complexity="high",

            training_cost="medium",

            supports_sparse_input=True,
            handles_missing_values=True,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=False,

            hyperparameter_space=xgboost_space,
            tie_breaker_score=1
        ),
        "dummy": ModelSpec(
            name="Dummy Regressor",
            factory=lambda: DummyRegressor(),

            family="baseline",

            requires_scaling=False,
            sensitive_to_dataset_size=False,

            captures_feature_interactions=False,
            feature_engineering_impact="low",
            prefers_log_transformed_features=False,
            supports_nonlinearity=False,

            tuning_complexity="low",

            training_cost="low",

            supports_sparse_input=True,
            handles_missing_values=True,

            scales_to_large_datasets=True,
            suitable_for_small_datasets=True,

            hyperparameter_space=empty_space,
            tie_breaker_score=99
        )
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
