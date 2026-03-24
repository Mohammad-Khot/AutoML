import pandas as pd
import numpy as np

from automl_engine.planning.config import DataQualityConfig
from automl_engine.data.leakage import apply_leakage_policy

# ─────────────── Generate Data ───────────────

np.random.seed(42)
n = 100

y = pd.Series(np.random.randint(0, 2, size=n), name="target")

X = pd.DataFrame({
    # 🔴 Exact duplicate of target
    "target_copy": y,

    # 🔴 Near-perfect linear correlation
    "linear_leak": y * 1.0,

    # 🔴 Monotonic relation (same ordering)
    "monotonic_leak": y + np.random.normal(0, 1e-6, size=n),

    # 🔴 ID column (high uniqueness)
    "user_id": [f"id_{i}" for i in range(n)],

    # 🔴 Temporal keyword
    "created_at": pd.date_range("2024-01-01", periods=n, freq="D"),

    # 🟢 Normal feature (should NOT trigger)
    "random_feature": np.random.randn(n),
})

# ─────────────── Config ───────────────

dq_config = DataQualityConfig(
    id_threshold=0.9,  # ensures user_id is flagged
    leak_handling="warn"  # so warnings are triggered
)

# ─────────────── Run Detection ───────────────

X_clean, leaks = apply_leakage_policy(X, y, dq_config)

print("\nReturned leaks:")
for col, reason in leaks:
    print(f"{col}: {reason}")
