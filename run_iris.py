# run_iris.py

from sklearn.datasets import load_iris
import pandas as pd
from automl_engine.core.engine import AutoMLEngine
from automl_engine.core.config import AutoMLConfig

X, y = load_iris(as_frame=True, return_X_y=True)
df = X.copy()
df["target"] = y

df.to_csv("data/iris.csv", index=False)

config = AutoMLConfig(
    metric="accuracy",
    seed=42
)

engine = AutoMLEngine(config)
trained_pipeline, scores = engine.run(
    "data/iris.csv",
    save_dir="artifacts/iris_run"
)

print("Saved model to artifacts/iris_run/")


print("Final scores:", scores)
# print("Trained pipeline:", trained_pipeline)