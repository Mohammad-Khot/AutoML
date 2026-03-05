# run_automl_benchmark.py

"""
AutoML Benchmark Harness

Runs your AutoML engine across many datasets automatically.

Features
--------
- 30+ OpenML benchmark datasets
- multiple random seeds
- failure logging
- runtime tracking
- reproducibility checks
- CSV report generation
"""

import time
import traceback
from pathlib import Path

import pandas as pd
import openml
from tqdm import tqdm
from joblib import Parallel, delayed

from automl_engine import AutoMLEngine, AutoMLConfig


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OUTPUT_DIR = Path("automl_benchmark")
DATA_DIR = OUTPUT_DIR / "datasets"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

DATASET_LIMIT = 30
SEEDS = [0, 1, 2, 42]

N_JOBS = 1   # increase to CPU count if desired


# --------------------------------------------------
# Download dataset
# --------------------------------------------------

def download_dataset(dataset_id):

    dataset = openml.datasets.get_dataset(dataset_id)

    X, y, categorical, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )

    df = X.copy()
    df["target"] = y

    path = DATA_DIR / f"{dataset.name}.csv"
    df.to_csv(path, index=False)

    return dataset.name, path


# --------------------------------------------------
# Fetch OpenML benchmark suite
# --------------------------------------------------

def build_dataset_suite(limit=30):

    print("\nDownloading OpenML benchmark datasets...\n")

    benchmark = openml.study.get_suite(99)  # OpenML-CC18
    dataset_ids = benchmark.data[:limit]

    datasets = []

    for did in tqdm(dataset_ids):

        try:
            name, path = download_dataset(did)
            datasets.append((name, path))

        except Exception as e:
            print("Dataset download failed:", did, e)

    return datasets


# --------------------------------------------------
# Run AutoML on one dataset
# --------------------------------------------------

def run_single_dataset(dataset_name, path, seed):

    start = time.time()

    try:

        config = AutoMLConfig(
            seed=seed,
            nested_cv=True,
            show_optuna_plots=False
        )

        engine = AutoMLEngine(config)

        engine.fit_from_path(path)

        summary = engine.summary()

        runtime = time.time() - start

        return {
            "dataset": dataset_name,
            "seed": seed,
            "status": "SUCCESS",
            "runtime_sec": runtime,
            "summary": str(summary)
        }

    except Exception as e:

        runtime = time.time() - start

        return {
            "dataset": dataset_name,
            "seed": seed,
            "status": "FAILED",
            "runtime_sec": runtime,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# --------------------------------------------------
# Run full benchmark
# --------------------------------------------------

def run_benchmark():

    datasets = build_dataset_suite(DATASET_LIMIT)

    jobs = []

    for name, path in datasets:
        for seed in SEEDS:
            jobs.append((name, path, seed))

    print("\nRunning AutoML benchmark...\n")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_single_dataset)(name, path, seed)
        for name, path, seed in tqdm(jobs)
    )

    df = pd.DataFrame(results)

    results_file = OUTPUT_DIR / "benchmark_results.csv"
    df.to_csv(results_file, index=False)

    print("\nBenchmark finished.")
    print("Results saved to:", results_file)

    summarize_results(df)


# --------------------------------------------------
# Basic analysis
# --------------------------------------------------

def summarize_results(df):

    print("\n===== BENCHMARK SUMMARY =====\n")

    total = len(df)
    failures = df[df.status == "FAILED"]

    print("Total runs:", total)
    print("Failures:", len(failures))
    print("Success rate:", round((total - len(failures)) / total * 100, 2), "%")

    if len(failures) > 0:
        print("\nFailed datasets:")
        print(failures[["dataset", "seed", "error"]].head())


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    run_benchmark()
