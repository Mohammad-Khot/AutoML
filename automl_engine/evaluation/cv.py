# automl_engine/evaluation/cv.py

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit
import warnings


def get_cv_object(task: str, y, folds: int, seed: int, strategy: str = "auto", repeats: int = 3):
    if task == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least 2 classes.")

    if strategy == "auto":
        strategy = "stratified" if task == "classification" else "kfold"

    if strategy == "stratified":
        min_class = y.value_counts().min()
        safe_folds = max(2, min(folds, min_class))

        if safe_folds != folds:
            warnings.warn(
                f"Reducing CV folds from {folds} to {safe_folds} "
                "due to small class size."
            )

        return StratifiedKFold(
            n_splits=safe_folds,
            shuffle=True,
            random_state=seed
        )

    if strategy == "kfold":
        return KFold(
            n_splits=max(2, folds),
            shuffle=True,
            random_state=seed
        )

    if strategy == "repeated":
        if task == "classification":
            return RepeatedStratifiedKFold(
                n_splits=folds,
                n_repeats=repeats,
                random_state=seed
            )
        return RepeatedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=seed
        )

    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=max(2, folds))

    raise ValueError(f"Unknown CV strategy : {strategy}")
