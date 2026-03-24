import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.datasets import make_classification, make_regression

from automl_engine import AutoMLEngine, AutoMLConfig

# ───────────────────────────────────────────────
# CONFIG SPACE
# ───────────────────────────────────────────────
TASKS = ["classification", "regression"]
SCALING = ["auto", "force", "none"]
ENCODING = ["auto", "onehot", "ordinal", "none"]
FOLDS = [3, 5]
SEEDS = [42, 7]


# ───────────────────────────────────────────────
# DATA GENERATORS
# ───────────────────────────────────────────────
def get_dataset(task, kind="normal", seed=42):
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
            X = np.random.randn(500, 10)
            y = np.random.randint(0, 2, 500)
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
            X = np.random.randn(500, 10)
            y = np.random.randn(500)
        else:
            X, y = make_regression(
                n_samples=500,
                n_features=10,
                noise=10.0,
                random_state=seed,
            )

    return pd.DataFrame(X), pd.Series(y)


# ───────────────────────────────────────────────
# CORE TEST RUNNER
# ───────────────────────────────────────────────
def run_engine(df, scaling, encoding, folds, seed):
    config = AutoMLConfig()

    config.runtime.seed = seed
    config.cv.folds = folds
    config.preprocessing.scaling = scaling
    config.preprocessing.encoding = encoding

    engine = AutoMLEngine(config)
    engine.fit(df)

    score = getattr(engine, "best_score_", None)
    model = getattr(engine, "best_model_name_", None)

    return score, model


def run_test(config_tuple, test_type, dataset_kind):
    task, scaling, encoding, folds, seed = config_tuple

    X, y = get_dataset(task, dataset_kind, seed)

    df = X.copy()
    df["target"] = y

    start = time.time()

    try:
        score, model = run_engine(df, task, scaling, encoding, folds, seed)
        status = "success"

    except Exception as e:
        score, model = None, None
        status = f"fail: {str(e)[:100]}"

    duration = time.time() - start

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
        "duration": duration,
    }


# ───────────────────────────────────────────────
# ALL TESTS
# ───────────────────────────────────────────────
def run_all_tests():
    results = []

    configs = [
        (t, s, e, f, seed)
        for t in TASKS
        for s in SCALING
        for e in ENCODING
        for f in FOLDS
        for seed in SEEDS
    ]

    # 1. STRUCTURAL
    for cfg in configs:
        results.append(run_test(cfg, "structural", "normal"))

    # 2. DETERMINISM
    for cfg in configs[:10]:
        r1 = run_test(cfg, "determinism_1", "normal")
        r2 = run_test(cfg, "determinism_2", "normal")

        # compare scores
        if r1["score"] == r2["score"]:
            r1["status"] = r2["status"] = "deterministic"
        else:
            r1["status"] = r2["status"] = "non_deterministic"

        results.extend([r1, r2])

    # 3. CORRECTNESS
    for cfg in configs[:10]:
        results.append(run_test(cfg, "perfect_data", "perfect"))
        results.append(run_test(cfg, "noise_data", "noise"))

    # 4. LEAKAGE
    for cfg in configs[:10]:
        task, scaling, encoding, folds, seed = cfg

        X, y = get_dataset(task, "normal", seed)

        df = X.copy()
        df["leak"] = y
        df["target"] = y

        try:
            score, model = run_engine(df, task, scaling, encoding, folds, seed)

            if task == "classification" and score is not None and score > 0.95:
                status = "leakage_detected"
            elif task == "regression" and score is not None and score < 1e-3:
                status = "leakage_detected"
            else:
                status = "no_leakage_signal"

        except Exception as e:
            score, model = None, None
            status = f"fail: {str(e)[:100]}"

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

    # 5. EDGE CASES
    for task in TASKS:
        X = pd.DataFrame(np.random.randn(100, 1))

        if task == "classification":
            y = pd.Series(np.random.randint(0, 2, 100))
        else:
            y = pd.Series(np.random.randn(100))

        df = X.copy()
        df["target"] = y

        try:
            score, model = run_engine(df, task, "auto", "auto", 3, 42)
            status = "success"

        except Exception as e:
            score, model = None, None
            status = f"fail: {str(e)[:100]}"

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


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
if __name__ == "__main__":
    df = run_all_tests()

    output_file = "../automl_test_results.csv"
    df.to_csv(output_file, index=False)

    print("\n🔥 TEST SUITE COMPLETE")
    print(f"Saved results → {output_file}")

    print("\nSummary:")
    print(df["status"].value_counts())
