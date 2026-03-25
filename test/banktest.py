import pandas as pd

from automl_engine import AutoMLConfig, AutoMLEngine

config = AutoMLConfig()

config.fast()
config.models.exclude_models = ["logistic_regression", "svc"]
config.cv.folds = 3
config.problem.metric = "f1"
engine = AutoMLEngine(config)

df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\bank-full.csv", sep=";")

engine.fit(df)
engine.summary()
