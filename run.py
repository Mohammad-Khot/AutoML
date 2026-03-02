# run.py
from automl_engine import AutoMLEngine, AutoMLConfig


def main():

    config = AutoMLConfig(
        seed=42,
        nested_cv=True
    )

    engine = AutoMLEngine(config)

    # ---- Train ----
    engine.fit_from_path("data/california.csv")

    # ---- Leaderboard ----
    engine.summary()


if __name__ == "__main__":
    main()
