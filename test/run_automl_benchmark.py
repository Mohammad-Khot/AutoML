from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import openml
from tqdm import tqdm

from automl_engine import AutoMLEngine, AutoMLConfig

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "automl_benchmark"
DATA_DIR = OUTPUT_DIR / "datasets"
RESULTS_DIR = OUTPUT_DIR / "per_dataset_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_LIMIT = 30
SEEDS = [0, 42, 570]


def download_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)

    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )

    df = X.copy()
    df["target"] = y

    path = DATA_DIR / f"{dataset.name}.csv"
    df.to_csv(path, index=False)

    return dataset.name, path


def build_dataset_suite(limit=30):
    suite = openml.study.get_suite(99)
    datasets = []

    for did in tqdm(suite.data[:limit], desc="Downloading datasets"):
        try:
            datasets.append(download_dataset(did))
        except Exception as exc:
            print("Download failed:", did, exc)

    return datasets


def extract_results(engine: AutoMLEngine):
    session = engine.session
    best_model = engine.best_model_name_
    best_score = engine.best_score_

    outer_mean = None
    outer_std = None
    if session.outer_scores:
        outer_mean = float(np.mean(session.outer_scores))
        outer_std = float(np.std(session.outer_scores))

    return {
        "best_model": best_model,
        "score": best_score,
        "outer_mean": outer_mean,
        "outer_std": outer_std,
        "runtime": engine._runtime,
    }


def run_single_dataset(dataset_name, path, seed):
    try:
        config = AutoMLConfig()
        config.runtime.seed = seed
        config.runtime.log = False
        config.cv.use_nested_cv = False
        config.search.compute_budget = "high"
        config.generate_optuna_plots = False
        config.display_optuna_plots = False
        config.optuna.enabled = False

        engine = AutoMLEngine(config)
        engine.fit(path)
        results = extract_results(engine)

        return {
            "dataset": dataset_name,
            "seed": seed,
            "status": "SUCCESS",
            "best_model": results["best_model"],
            "score": results["score"],
            "outer_mean": results["outer_mean"],
            "outer_std": results["outer_std"],
            "runtime_sec": results["runtime"],
        }

    except Exception as exc:
        return {
            "dataset": dataset_name,
            "seed": seed,
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_benchmark():
    datasets = build_dataset_suite(DATASET_LIMIT)
    results_file = OUTPUT_DIR / "benchmark_results.csv"

    for dataset_name, path in datasets:
        dataset_results = []

        for seed in SEEDS:
            print(f"\nRunning {dataset_name} | seed={seed}")
            dataset_results.append(run_single_dataset(dataset_name, path, seed))

        pd.DataFrame(dataset_results).to_csv(
            RESULTS_DIR / f"{dataset_name}.csv",
            index=False,
        )

    all_results = [pd.read_csv(file) for file in RESULTS_DIR.glob("*.csv")]
    if not all_results:
        raise RuntimeError("No benchmark results were generated.")

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(results_file, index=False)

    print("\nBenchmark complete")
    print("Results saved:", results_file)
    summarize(final_df)


def summarize(df):
    total = len(df)
    failures = df[df.status == "FAILED"]

    print("\n===== BENCHMARK SUMMARY =====")
    print("Total runs:", total)
    print("Failures:", len(failures))
    print("Success rate:", round((total - len(failures)) / total * 100, 2), "%")

    if len(failures) > 0:
        print("\nFailed runs:")
        print(failures[["dataset", "seed", "error"]])


if __name__ == "__main__":
    run_benchmark()
