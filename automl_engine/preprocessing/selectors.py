from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_regression,
    mutual_info_classif,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression


def safe_k(total_features, k):
    return max(1, min(k, total_features))


def get_selector(task, mode="auto", n_features=None):
    if mode == "none" or not n_features:
        return "passthrough"

    if mode == "auto":
        k = safe_k(n_features, 20)
        if k >= n_features:
            return "passthrough"

        if task == "classification":
            return SelectKBest(mutual_info_classif, k=20)
        else:
            return SelectKBest(mutual_info_regression, k=20)

    if mode == "variance":
        return VarianceThreshold(threshold=0.0)

    if mode == "l1":
        if n_features < 5:
            return "passthrough"

        if task == "classification":
            base = LogisticRegression(penalty="l1", solver="liblinear")
        else:
            base = Lasso(alpha=0.1)
        return SelectFromModel(base)

    if mode == "tree":
        if n_features < 3:
            return "passthrough"

        if task == "classification":
            base = RandomForestClassifier(n_estimators=100)
        else:
            base = RandomForestRegressor(n_estimators=100)
        return SelectFromModel(base)

    raise ValueError(f"Unknown selector mode : {mode}")
