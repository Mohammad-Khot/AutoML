import pandas as pd


# ───────────────────────────────────────────────
# 🔥 HARD TRAP: catch ANY hashing attempt on Series
# ───────────────────────────────────────────────
def crash_hash(self):
    raise RuntimeError("🔥 SERIES HASH CALLED HERE 🔥")


pd.Series.__hash__ = crash_hash

import traceback
from datetime import datetime
from itertools import product

from sklearn.datasets import make_classification, make_regression

# Import your engine + config
from automl_engine import AutoMLEngine
from automl_engine.planning.config import (
    AutoMLConfig,
    RuntimeConfig,
    CVConfig,
    PreprocessingConfig,
)

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────
LOG_FILE = "../crash_test_results.txt"


def log_to_file(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


# ───────────────────────────────────────────────
# Dataset Generator
# ───────────────────────────────────────────────
def generate_dataset(task: str):
    if task == "classification":
        X, y = make_classification(
            n_samples=5000,
            n_features=20,
            n_informative=10,
            n_classes=2,
            random_state=42,
        )
    else:
        X, y = make_regression(
            n_samples=5000,
            n_features=20,
            noise=0.1,
            random_state=42,
        )

    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")

    return X, y


# ───────────────────────────────────────────────
# Config Space
# ───────────────────────────────────────────────
TASKS = ["classification", "regression"]
SCALING = ["auto", "force", "none"]
ENCODING = ["auto", "onehot", "ordinal", "none"]
CV_FOLDS = [3, 5]
SEEDS = [42, 7]


# ───────────────────────────────────────────────
# Crash Test Runner
# ───────────────────────────────────────────────
def run_test():
    total_runs = 0
    failures = 0

    # reset log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("🔥 AutoML Crash Test Log\n\n")

    for task, scaling, encoding, folds, seed in product(
            TASKS, SCALING, ENCODING, CV_FOLDS, SEEDS
    ):
        total_runs += 1

        header = (
                "\n" + "=" * 60 + "\n"
                                  f"RUN #{total_runs}\n"
                                  f"task={task}, scaling={scaling}, encoding={encoding}, folds={folds}, seed={seed}\n"
                + "=" * 60
        )

        print(header)
        log_to_file(header)

        try:
            X, y = generate_dataset(task)

            config = AutoMLConfig(
                runtime=RuntimeConfig(
                    log=False,
                    seed=seed,
                ),
                cv=CVConfig(
                    folds=folds,
                ),
                preprocessing=PreprocessingConfig(
                    scaling_mode=scaling,
                    encoding_strategy=encoding,
                ),
            )

            engine = AutoMLEngine(user_config=config)

            start = datetime.now()
            engine.fit((X, y))  # ✅ correct usage
            end = datetime.now()

            result = f"✅ SUCCESS ({(end - start).total_seconds():.2f}s)"
            print(result)
            log_to_file(result)

        except Exception:
            failures += 1

            print("\n🔥 CRASH DETECTED 🔥")
            log_to_file("🔥 CRASH DETECTED 🔥")

            tb = traceback.format_exc()
            print(tb)
            log_to_file(tb)

        # 🔁 flush after each run (important for live tracking)
        import sys
        sys.stdout.flush()

    summary = (
            "\n" + "#" * 60 + "\n"
                              f"TOTAL RUNS: {total_runs}\n"
                              f"FAILURES: {failures}\n"
            + "#" * 60
    )

    print(summary)
    log_to_file(summary)


# ───────────────────────────────────────────────
# Entry Point
# ───────────────────────────────────────────────
if __name__ == "__main__":
    run_test()
