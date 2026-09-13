import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from automl_engine import AutoMLEngine, AutoMLConfig

TASKS = ["classification", "regression"]
SCALING = ["auto", "force", "none"]
ENCODING = ["auto", "onehot", "ordinal", "none"]
FOLDS = [3, 5]
SEEDS = [42, 7]


def get_dataset(task, kind="normal", seed=42):
    rng = np.random.default_rng(seed)

    if task == "classification":
        if kind == "perfect":
            X, y = make_classification(
                n_samples=500,
                n_features=10,
                n_informative=10,
                n_redundant=0,
                class_sep=5.0,
                random_state=seed,
            )
        elif kind == "noise":
            X = rng.normal(size=(500, 10))
            y = rng.integers(0, 2, 500)
        else:
            X, y = make_classification(
                n_samples=500,
                n_features=10,
                n_informative=5,
                random_state=seed,
            )
    else:
        if kind == "perfect":
            X, y = make_regression(
                n_samples=500,
                n_features=10,
                noise=0.0,
                random_state=seed,
            )
        elif kind == "noise":
            X = rng.normal(size=(500, 10))
            y = rng.normal(size=500)
        else:
            X, y = make_regression(
                n_samples=500,
                n_features=10,
                noise=10.0,
                random_state=seed,
            )

    return pd.DataFrame(X), pd.Series(y)


def run_engine(df, task, scaling, encoding, folds, seed):
    config = AutoMLConfig()
    config.problem.task = task
    config.runtime.seed = seed
    config.runtime.log = False
    config.cv.folds = folds
    config.preprocessing.scaling_mode = scaling
    config.preprocessing.encoding_strategy = encoding

    # This suite exercises model selection rather than expensive tuning.
    config.optuna.enabled = False
    config.generate_optuna_plots = False

    engine = AutoMLEngine(config)
    engine.fit(df)

    return engine.best_score_, engine.best_model_name_


def run_test(config_tuple, test_type, dataset_kind):
    task, scaling, encoding, folds, seed = config_tuple
    X, y = get_dataset(task, dataset_kind, seed)

    df = X.copy()
    df["target"] = y

    start = time.time()

    try:
        score, model = run_engine(df, task, scaling, encoding, folds, seed)
        status = "success"
    except Exception as exc:
        score, model = None, None
        status = f"fail: {str(exc)[:100]}"

    return {
        "timestamp": datetime.now(),
        "test_type": test_type,
        "dataset_kind": dataset_kind,
        "task": task,
        "scaling": scaling,
        "encoding": encoding,
        "folds": folds,
        "seed": seed,
        "status": status,
        "score": score,
        "model": model,
        "duration": time.time() - start,
    }


def run_all_tests():
    results = []

    configs = [
        (task, scaling, encoding, folds, seed)
        for task in TASKS
        for scaling in SCALING
        for encoding in ENCODING
        for folds in FOLDS
        for seed in SEEDS
    ]

    for cfg in configs:
        results.append(run_test(cfg, "structural", "normal"))

    for cfg in configs[:10]:
        r1 = run_test(cfg, "determinism_1", "normal")
        r2 = run_test(cfg, "determinism_2", "normal")

        if r1["status"] == "success" and r2["status"] == "success":
            deterministic = np.isclose(r1["score"], r2["score"], equal_nan=True)
            r1["status"] = r2["status"] = (
                "deterministic" if deterministic else "non_deterministic"
            )

        results.extend([r1, r2])

    for cfg in configs[:10]:
        results.append(run_test(cfg, "perfect_data", "perfect"))
        results.append(run_test(cfg, "noise_data", "noise"))

    for cfg in configs[:10]:
        task, scaling, encoding, folds, seed = cfg
        X, y = get_dataset(task, "normal", seed)

        df = X.copy()
        df["leak"] = y
        df["target"] = y

        try:
            score, model = run_engine(df, task, scaling, encoding, folds, seed)

            if task == "classification" and score > 0.95:
                status = "leakage_detected"
            elif task == "regression" and score > 0.95:
                status = "leakage_detected"
            else:
                status = "no_leakage_signal"
        except Exception as exc:
            score, model = None, None
            status = f"fail: {str(exc)[:100]}"

        results.append({
            "timestamp": datetime.now(),
            "test_type": "leakage",
            "dataset_kind": "leak",
            "task": task,
            "scaling": scaling,
            "encoding": encoding,
            "folds": folds,
            "seed": seed,
            "status": status,
            "score": score,
            "model": model,
            "duration": None,
        })

    for task in TASKS:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(100, 1)))
        y = pd.Series(
            rng.integers(0, 2, 100)
            if task == "classification"
            else rng.normal(size=100)
        )

        df = X.copy()
        df["target"] = y

        try:
            score, model = run_engine(df, task, "auto", "auto", 3, 42)
            status = "success"
        except Exception as exc:
            score, model = None, None
            status = f"fail: {str(exc)[:100]}"

        results.append({
            "timestamp": datetime.now(),
            "test_type": "edge_case",
            "dataset_kind": "minimal",
            "task": task,
            "scaling": None,
            "encoding": None,
            "folds": None,
            "seed": None,
            "status": status,
            "score": score,
            "model": model,
            "duration": None,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_all_tests()

    output_file = Path(__file__).resolve().parent.parent / "automl_test_results.csv"
    df.to_csv(output_file, index=False)

    print("\nTEST SUITE COMPLETE")
    print(f"Saved results -> {output_file}")
    print("\nSummary:")
    print(df["status"].value_counts())
