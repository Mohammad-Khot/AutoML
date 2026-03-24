from pathlib import Path
import traceback
import pandas as pd
import openml
from tqdm import tqdm

from automl_engine import AutoMLEngine, AutoMLConfig

# --------------------------------------------------
# Benchmark settings
# --------------------------------------------------

OUTPUT_DIR = Path("../automl_benchmark")
DATA_DIR = OUTPUT_DIR / "datasets"
RESULTS_DIR = OUTPUT_DIR / "per_dataset_results"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATASET_LIMIT = 30
SEEDS = [0, 42, 570]


# --------------------------------------------------
# Download OpenML dataset
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
# Build dataset suite
# --------------------------------------------------

def build_dataset_suite(limit=30):
    suite = openml.study.get_suite(99)

    dataset_ids = suite.data[:limit]

    datasets = []

    for did in tqdm(dataset_ids, desc="Downloading datasets"):

        try:
            name, path = download_dataset(did)
            datasets.append((name, path))

        except Exception as e:
            print("Download failed:", did, e)

    return datasets


# --------------------------------------------------
# Extract results from engine session
# --------------------------------------------------

def extract_results(engine: AutoMLEngine):
    session = engine.session_

    best_model = session.best_model_name
    scores = session.search_state.scores

    best_score = scores.get(best_model)

    outer = engine.outer_summary()

    outer_mean = None
    outer_std = None

    if outer is not None:
        outer_mean = outer["mean"]
        outer_std = outer["std"]

    return {
        "best_model": best_model,
        "score": best_score,
        "outer_mean": outer_mean,
        "outer_std": outer_std,
        "runtime": engine._runtime,
    }


# --------------------------------------------------
# Run engine once
# --------------------------------------------------

def run_single_dataset(dataset_name, path, seed):
    try:

        config = AutoMLConfig(
            seed=seed,
            task=None,
            metric=None,
            log=False,
            show_optuna_plots=False,
            return_optuna_plots=False,
            nested_cv=False,
            max_compute="high",
        )

        engine = AutoMLEngine(config)

        engine.fit_from_path(path)

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

    except Exception as e:

        return {
            "dataset": dataset_name,
            "seed": seed,
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# --------------------------------------------------
# Run benchmark
# --------------------------------------------------

def run_benchmark():
    datasets = build_dataset_suite(DATASET_LIMIT)

    results_file = OUTPUT_DIR / "benchmark_results.csv"

    for dataset_name, path in datasets:

        dataset_results = []

        for seed in SEEDS:
            print(f"\nRunning {dataset_name} | seed={seed}")

            result = run_single_dataset(dataset_name, path, seed)

            dataset_results.append(result)

        # ------------------------------------------
        # SAVE RESULTS FOR THIS DATASET
        # ------------------------------------------

        dataset_df = pd.DataFrame(dataset_results)

        dataset_file = RESULTS_DIR / f"{dataset_name}.csv"

        dataset_df.to_csv(dataset_file, index=False)

        print(f"Saved dataset results: {dataset_file}")

    # ------------------------------------------
    # CONCATENATE ALL RESULTS
    # ------------------------------------------

    all_files = list(RESULTS_DIR.glob("*.csv"))

    all_results = []

    for file in all_files:
        df = pd.read_csv(file)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    final_df.to_csv(results_file, index=False)

    print("\nBenchmark complete")
    print("Results saved:", results_file)

    summarize(final_df)


# --------------------------------------------------
# Benchmark summary
# --------------------------------------------------

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


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
