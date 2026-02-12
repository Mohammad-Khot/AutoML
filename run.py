# run.py
from automl_engine import AutoMLEngine, AutoMLConfig
import numpy as np
import time


def print_scores(scores):
    print("\n=== Model Performance (Inner CV) ===")

    for name, value in sorted(
            scores["inner_scores"].items(),
            key=lambda x: float(x[1]),
            reverse=True
    ):
        print(f"{name:<14} → {float(value):>6.4f}")

    if scores.get("outer_scores"):
        raw = scores["outer_scores"]

        # Keep only things that look numeric
        outer = []
        bad = []

        for s in raw:
            try:
                outer.append(float(s))
            except (TypeError, ValueError):
                bad.append(s)

        if bad:
            print(f"\nWarning: ignored non-numeric outer values → {bad}")

        if not outer:
            print("\nNo valid outer scores to display")
            return

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
    trained_pipeline, scores = engine.run("data/iris.csv")

    # print("\nPipeline:", trained_pipeline)

    print_scores(scores)

    elapsed = time.perf_counter() - start
    print(f"\nTime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
