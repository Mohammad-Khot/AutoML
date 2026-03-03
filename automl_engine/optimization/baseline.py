# optimization/baseline.py

def run_baseline(pipeline, X, y):
    pipeline.fit(X, y)
    return pipeline
