# load_and_predict.py
from automl_engine.utils.persistence import load_pipeline
import pandas as pd

model = load_pipeline("artifacts/iris_run/model.joblib")

df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target"])

preds = model.predict(X)
print(preds[:5])
