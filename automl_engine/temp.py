from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def print_pipeline(pipeline, indent=0):
    space = " " * indent

    # ───────── PIPELINE ─────────
    if isinstance(pipeline, Pipeline):
        print(f"{space}Pipeline:")
        for name, step in pipeline.steps:
            print(f"{space}  ├── {name}: {step.__class__.__name__}")
            print_pipeline(step, indent + 6)

    # ───────── COLUMN TRANSFORMER ─────────
    elif isinstance(pipeline, ColumnTransformer):
        print(f"{space}ColumnTransformer:")
        for name, transformer, cols in pipeline.transformers:
            if transformer == "drop":
                continue

            print(f"{space}  ├── {name} ({list(cols)}): {transformer.__class__.__name__}")
            print_pipeline(transformer, indent + 6)

    # ───────── SINGLE ESTIMATOR ─────────
    else:
        # Leaf node (actual transformer/model)
        try:
            params = pipeline.get_params(deep=False)
            clean_params = {
                k: v for k, v in params.items()
                if isinstance(v, (int, float, str, bool))
            }
        except:
            clean_params = {}

        if clean_params:
            print(f"{space}      params: {clean_params}")
