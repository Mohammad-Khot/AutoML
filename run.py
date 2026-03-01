# run.py
from automl_engine import AutoMLEngine, AutoMLConfig
import time


def main():
    start = time.perf_counter()

    config = AutoMLConfig(
        seed=42,
        nested_cv=True
    )

    engine = AutoMLEngine(config)

    # ---- Train ----
    engine.fit_from_path("data/california.csv")

    # ---- Leaderboard ----
    engine.summary()

    elapsed = time.perf_counter() - start
    print(f"\nTime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
