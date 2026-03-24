# run_multi_dataset.py

from pathlib import Path
import pandas as pd

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_california_housing,
    make_regression,
)

from automl_engine import AutoMLEngine, AutoMLConfig
from automl_engine.reporting.console import CONSOLE_WIDTH

DATA_DIR = Path("../temp_datasets")
DATA_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Utility: save sklearn dataset as CSV
# --------------------------------------------------
def save_dataset(X, y, feature_names, name: str):
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)

    return path


# --------------------------------------------------
# Dataset builders
# --------------------------------------------------
def build_datasets():
    datasets = []

    # 1️⃣ Iris — small multiclass classification
    iris = load_iris()
    datasets.append(
        save_dataset(
            iris.data,
            iris.target,
            iris.feature_names,
            "iris"
        )
    )

    # 2️⃣ Wine — medium classification
    wine = load_wine()
    datasets.append(
        save_dataset(
            wine.data,
            wine.target,
            wine.feature_names,
            "wine"
        )
    )

    # 3️⃣ Breast Cancer — binary classification
    cancer = load_breast_cancer()
    datasets.append(
        save_dataset(
            cancer.data,
            cancer.target,
            cancer.feature_names,
            "breast_cancer"
        )
    )

    # 4️⃣ California Housing — real regression dataset
    housing = fetch_california_housing()
    datasets.append(
        save_dataset(
            housing.data,
            housing.target,
            housing.feature_names,
            "california"
        )
    )

    # 5️⃣ Synthetic Regression (noise test)
    X, y = make_regression(
        n_samples=2000,
        n_features=25,
        noise=15,
        random_state=42
    )

    feature_names = [f"f{i}" for i in range(X.shape[1])]
    datasets.append(
        save_dataset(
            X,
            y,
            feature_names,
            "synthetic_regression"
        )
    )

    return datasets


# --------------------------------------------------
# Run AutoML on all datasets
# --------------------------------------------------
def main():
    config = AutoMLConfig()

    config.runtime.seed = 42
    config.cv.use_nested_cv = True
    config.display_optuna_plots = False
    config.generate_optuna_plots = False
    # config.models.include_models = ["logistic"]
    paths = build_datasets()

    for path in paths:
        print("\n\n" + "=" * CONSOLE_WIDTH + "\n\n")
        print(f"RUNNING AUTO ML ON: {path.name}")
        print("\n\n" + "=" * CONSOLE_WIDTH)

        engine = AutoMLEngine(config)

        engine.fit(path)
        engine.summary()


if __name__ == "__main__":
    main()
    