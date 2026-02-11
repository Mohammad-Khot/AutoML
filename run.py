# run.py
from automl_engine import AutoMLEngine, AutoMLConfig
import numpy as np
import time


def print_scores(scores):
    print("\n=== Model Performance (Inner CV) ===")

    for name, value in sorted(scores["inner_scores"].items(),
                              key=lambda x: x[1],
                              reverse=True):
        print(f"{name:<14} → {value:>6.4f}")

    if scores["outer_scores"]:
        outer = scores["outer_scores"]
        print("\n=== Outer CV ===")
        print(f"folds : {[round(s, 4) for s in outer]}")
        print(f"mean  : {np.mean(outer):.4f}")
        print(f"std   : {np.std(outer):.4f}")


def main():
    start = time.perf_counter()

    config = AutoMLConfig(
        seed=42
    )

    engine = AutoMLEngine(config)
    trained_pipeline, scores = engine.run("data/california.csv")

    # print("\nPipeline:", trained_pipeline)

    print_scores(scores)

    elapsed = time.perf_counter() - start
    print(f"\nTime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
