from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_selector(config):
    method = config.preprocessing.feature_selection_method

    if method in ("none", None):
        return "passthrough"

    task = config.problem.task

    if method == "variance":
        return VarianceThreshold(threshold=0.0)

    if method == "l1":
        if task == "classification":
            return SelectFromModel(
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=1.0
                )
            )
        else:
            return SelectFromModel(Lasso(alpha=0.01))

    if method == "tree":
        if task == "classification":
            return SelectFromModel(RandomForestClassifier(n_estimators=100))
        else:
            return SelectFromModel(RandomForestRegressor(n_estimators=100))

    return "passthrough"
