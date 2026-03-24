# test/run_dirty_dataset_full.py

import numpy as np
import pandas as pd
from pathlib import Path

from automl_engine import AutoMLEngine, AutoMLConfig


def generate_dirty_classification(n_samples: int = 500, seed: int = 42):
    np.random.seed(seed)

    # =========================================================
    # CORE SIGNAL FEATURES
    # =========================================================
    age = np.random.normal(40, 12, n_samples)
    income = np.random.normal(60000, 15000, n_samples)
    score = np.random.uniform(0, 100, n_samples)

    # =========================================================
    # CORRELATED FEATURES (useful)
    # =========================================================
    age_scaled = age * 1.1 + np.random.normal(0, 2, n_samples)
    income_log = np.log(income + 1)
    score_sqrt = np.sqrt(score + 1)

    # =========================================================
    # NOISE FEATURES (should be removed by selector)
    # =========================================================
    noise_1 = np.random.normal(0, 1, n_samples)
    noise_2 = np.random.normal(0, 1, n_samples)
    noise_3 = np.random.uniform(-1, 1, n_samples)
    noise_4 = np.random.uniform(-10, 10, n_samples)
    noise_5 = np.random.normal(100, 50, n_samples)

    # =========================================================
    # REDUNDANT / LINEAR COMBINATIONS
    # =========================================================
    income_per_age = income / (age + 1)
    score_per_income = score / (income + 1)
    age_income_interaction = age * income

    # =========================================================
    # OUTLIERS
    # =========================================================
    income[:10] *= 5

    # =========================================================
    # MISSING VALUES
    # =========================================================
    age[np.random.choice(n_samples, 50, replace=False)] = np.nan
    income[np.random.choice(n_samples, 30, replace=False)] = np.nan

    # =========================================================
    # CATEGORICAL FEATURES
    # =========================================================
    city = np.random.choice(["Mumbai", "Delhi", "Pune", "Chennai"], n_samples)
    device = np.random.choice(["Mobile", "Desktop", "Tablet"], n_samples)
    segment = np.random.choice(["A", "B", "C"], n_samples)

    city[np.random.choice(n_samples, 20, replace=False)] = None

    # =========================================================
    # TARGET (depends only on core features)
    # =========================================================
    y = (
            (age > 35).astype(int)
            + (income > 50000).astype(int)
            + (score > 50).astype(int)
    )
    y = (y >= 2).astype(int)

    df = pd.DataFrame({
        # core
        "age": age,
        "income": income,
        "score": score,

        # correlated
        "age_scaled": age_scaled,
        "income_log": income_log,
        "score_sqrt": score_sqrt,

        # interactions
        "income_per_age": income_per_age,
        "score_per_income": score_per_income,
        "age_income_interaction": age_income_interaction,

        # noise
        "noise_1": noise_1,
        "noise_2": noise_2,
        "noise_3": noise_3,
        "noise_4": noise_4,
        "noise_5": noise_5,

        # categorical
        "city": city,
        "device": device,
        "segment": segment,

        # target
        "target": y
    })

    return df


def main():
    base_dir = Path(__file__).parent
    save_path = base_dir / "dirty_dataset.csv"

    print("=" * 75)
    print("GENERATING DIRTY DATASET (HIGH-DIMENSION)")
    print("=" * 75)

    df = generate_dirty_classification()

    print(f"[DEBUG] Saving to: {save_path}")
    df.to_csv(save_path, index=False)

    if not save_path.exists():
        raise RuntimeError("Dataset not created")

    print(f"Dataset created successfully with {df.shape[1] - 1} features")

    print("=" * 75)
    print("RUNNING AUTO ML ON: dirty_dataset.csv")
    print("=" * 75)

    config = AutoMLConfig()
    config.cv.use_nested_cv = True

    engine = AutoMLEngine(config)
    engine.fit(save_path)


if __name__ == "__main__":
    main()
