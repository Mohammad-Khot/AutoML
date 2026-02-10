# automl_engine/evaluation/cv.py

from sklearn.model_selection import StratifiedKFold, KFold
import warnings


def get_cv(task, y, folds, seed):
    if task == "classification":
        min_class_count = y.value_counts().min()

        safe_folds = max(2, min(folds, min_class_count))

        if safe_folds != folds:
            warnings.warn(
                f"Reducing CV folds from {folds} to {safe_folds} "
                "due to small class size."
            )
            folds = min_class_count

        return StratifiedKFold(
            n_splits=safe_folds,
            shuffle=True,
            random_state=seed
        )

    safe_folds = max(2, folds)

    return KFold(
        n_splits=safe_folds,
        shuffle=True,
        random_state=seed
    )
