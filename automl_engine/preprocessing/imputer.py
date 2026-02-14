# preprocessing/imputer.py

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer


def select_imputer_strategy(X, config):
    if config.imputation == "none":
        return None, None

    num_strategy = None
    cat_strategy = None

    if config.imputation == "simple":
        num_strategy = SimpleImputer(strategy="median")
        cat_strategy = SimpleImputer(strategy="most_frequent")

    elif config.imputation == "knn":
        num_strategy = KNNImputer()

    elif config.imputation == "iterative":
        num_strategy = IterativeImputer(random_state=config.seed)

    elif config.imputation == "auto":
        if X.shape[0] < 2000:
            num_strategy = IterativeImputer(random_state=config.seed)
        elif X.shape[0] < 20_000:
            num_strategy = KNNImputer()
        else:
            num_strategy = SimpleImputer(strategy="median")

        cat_strategy = SimpleImputer(strategy="most_frequent")

    return num_strategy, cat_strategy
