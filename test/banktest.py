import pandas as pd

from automl_engine import AutoMLConfig, AutoMLEngine

config = AutoMLConfig()

config.fast()
config.models.exclude_models = ["logistic"]
config.cv.folds = 3
config.cv.use_nested_cv = True
engine = AutoMLEngine(config)

df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\bank-full.csv", sep=";")

engine.fit(df)
engine.summary()
